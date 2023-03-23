// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Float32Array, RecordBatch};
use arrow_schema::DataType::Float32;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use datafusion::execution::{
    context::SessionState,
    runtime_env::{RuntimeConfig, RuntimeEnv},
};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::{
    limit::GlobalLimitExec, ExecutionPlan, PhysicalExpr, SendableRecordBatchStream,
};
use datafusion::prelude::*;
use futures::stream::{Stream, StreamExt};

use super::Dataset;
use crate::datafusion::physical_expr::column_names_in_expr;
use crate::datatypes::Schema;
use crate::format::Index;
use crate::index::vector::{MetricType, Query, SCORE_COL};
use crate::io::exec::{
    GlobalTakeExec, KNNFlatExec, KNNIndexExec, LanceScanExec, LocalTakeExec, Planner,
};
use crate::utils::sql::parse_sql_filter;
use crate::{Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
pub const DEFAULT_BATCH_SIZE: usize = 8192;

const PREFETCH_SIZE: usize = 8;

/// Dataset Scanner
///
/// ```rust,ignore
/// let dataset = Dataset::open(uri).await.unwrap();
/// let stream = dataset.scan()
///     .project(&["col", "col2.subfield"]).unwrap()
///     .limit(10)
///     .into_stream();
/// stream
///   .map(|batch| batch.num_rows())
///   .buffered(16)
///   .sum()
/// ```
pub struct Scanner {
    dataset: Arc<Dataset>,

    /// Projection.
    projections: Schema,

    /// Optional filters string.
    filter: Option<String>,

    /// The batch size controls the maximum size of rows to return for each read.
    batch_size: usize,

    limit: Option<i64>,
    offset: Option<i64>,

    nearest: Option<Query>,

    /// Scan the dataset with a meta column: "_rowid"
    with_row_id: bool,
}

impl Scanner {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        let projection = dataset.schema().clone();
        Self {
            dataset,
            projections: projection,
            filter: None,
            batch_size: DEFAULT_BATCH_SIZE,
            limit: None,
            offset: None,
            nearest: None,
            with_row_id: false,
        }
    }

    /// Projection.
    ///
    /// Only seelect the specific columns. If not specifid, all columns will be scanned.
    pub fn project(&mut self, columns: &[&str]) -> Result<&mut Self> {
        self.projections = self.dataset.schema().project(columns)?;
        Ok(self)
    }

    /// Apply filters
    ///
    /// The filters can be presented as the string, as in WHERE clause in SQL.
    ///
    /// ```rust,ignore
    /// let dataset = Dataset::open(uri).await.unwrap();
    /// let stream = dataset.scan()
    ///     .project(&["col", "col2.subfield"]).unwrap()
    ///     .filter("a > 10 AND b < 200").unwrap()
    ///     .limit(10)
    ///     .into_stream();
    /// ```
    ///
    /// Once the filter is applied, Lance will create an optimized I/O plan for filtering.
    ///
    pub fn filter(&mut self, filter: &str) -> Result<&mut Self> {
        parse_sql_filter(filter)?;
        self.filter = Some(filter.to_string());
        Ok(self)
    }

    /// Set the batch size.
    pub fn batch_size(&mut self, batch_size: usize) -> &mut Self {
        self.batch_size = batch_size;
        self
    }

    /// Set limit and offset.
    pub fn limit(&mut self, limit: i64, offset: Option<i64>) -> Result<&mut Self> {
        if limit < 0 {
            return Err(Error::IO("Limit must be non-negative".to_string()));
        }
        if let Some(off) = offset {
            if off < 0 {
                return Err(Error::IO("Offset must be non-negative".to_string()));
            }
        }
        self.limit = Some(limit);
        self.offset = offset;
        Ok(self)
    }

    /// Find k-nearest neighbour within the vector column.
    pub fn nearest(&mut self, column: &str, q: &Float32Array, k: usize) -> Result<&mut Self> {
        if k == 0 {
            return Err(Error::IO("k must be positive".to_string()));
        }
        if q.is_empty() {
            return Err(Error::IO(
                "Query vector must have non-zero length".to_string(),
            ));
        }
        // make sure the field exists
        self.dataset.schema().project(&[column])?;
        self.nearest = Some(Query {
            column: column.to_string(),
            key: Arc::new(q.clone()),
            k,
            nprobes: 1,
            refine_factor: None,
            metric_type: MetricType::L2,
            use_index: true,
        });
        Ok(self)
    }

    pub fn nprobs(&mut self, n: usize) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.nprobes = n;
        }
        self
    }

    /// Apply a refine step to the vector search.
    ///
    /// A refine step uses the original vector values to re-rank the distances.
    pub fn refine(&mut self, factor: u32) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.refine_factor = Some(factor)
        };
        self
    }

    /// Change the distance [MetricType], i.e, L2 or Cosine distance.
    pub fn distance_metric(&mut self, metric_type: MetricType) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.metric_type = metric_type
        }
        self
    }

    /// Set whether to use the index if available
    pub fn use_index(&mut self, use_index: bool) -> &mut Self {
        if let Some(q) = self.nearest.as_mut() {
            q.use_index = use_index
        }
        self
    }

    /// Instruct the scanner to return the `_rowid` meta column from the dataset.
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    /// The Arrow schema of the output, including projections and vector / score.
    pub fn schema(&self) -> Result<SchemaRef> {
        let schema = if self.nearest.as_ref().is_some() {
            // If nearest neighbour search is enabled, we need to add the vector column and score column.
            let q = self.nearest.as_ref().unwrap();
            let vector_column: ArrowField = self
                .dataset
                .schema()
                .field(&q.column)
                .ok_or_else(|| Error::IO(format!("Column {} does not exist", q.column)))?
                .into();
            let score = ArrowField::new(SCORE_COL, Float32, false);
            let vector_extra_schema =
                Schema::try_from(&ArrowSchema::new(vec![vector_column, score]))?;
            self.projections.merge(&vector_extra_schema)
        } else {
            self.projections.clone()
        };
        let arrow_schema =
            ArrowSchema::try_from(&schema).map_err(|_| Error::IO("Invalid schema".to_string()))?;
        Ok(Arc::new(arrow_schema))
    }

    /// Create a stream of this Scanner.
    ///
    /// TODO: implement as IntoStream/IntoIterator.
    pub async fn try_into_stream(&self) -> Result<RecordBatchStream> {
        let with_row_id = self.with_row_id;
        let projection = &self.projections;

        let filter_expr = if let Some(filter) = self.filter.as_ref() {
            let planner = crate::io::exec::Planner::new(Arc::new(self.dataset.schema().into()));
            let logical_expr = planner.parse_filter(filter)?;
            Some(planner.create_physical_expr(&logical_expr)?)
        } else {
            None
        };

        let mut plan: Arc<dyn ExecutionPlan> = if self.nearest.is_some() {
            self.knn().await?
        } else if let Some(filter) = filter_expr {
            let columns_in_filter = column_names_in_expr(filter.as_ref());
            let filter_schema = Arc::new(
                self.dataset.schema().project(
                    &columns_in_filter
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                )?,
            );
            let scan = self.scan(true, filter_schema);
            let filter_node = self.filter_node(filter, scan, Arc::new(projection.clone()), true)?;
            self.take(filter_node, projection, true)
        } else {
            self.scan(with_row_id, Arc::new(self.projections.clone()))
        };

        if (self.limit.unwrap_or(0) > 0) || self.offset.is_some() {
            plan = self.limit_node(plan);
        }

        let session_config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
        let session_state = SessionState::with_config_rt(session_config, runtime_env);
        Ok(RecordBatchStream::new(
            plan.execute(0, session_state.task_ctx())?,
        ))
    }

    /// Create an execution plan for a Scan
    ///
    /// Simple scan:
    ///   Scan(projections)
    /// With filter and limit:
    ///   Scan(filter_columns) -> Filter() -> *(Limit) -> Take(other_columns)
    /// With KNN index:
    ///   KNNIndex -> Take(vector) -> FlatRefine -> *(Filter) -> *(Limit) -> Take(other_columns)
    /// With KNN flat:
    ///   Scan(vector_col) -> FlatKNN -> Take(filter_column)
    ///     -> *(Filter) -> *(Limit) -> Take(other_columns)
    ///
    /// In general, the execution plan includes the following steps:
    ///
    ///   1. Source stage: Scan from the source or from index if presented.
    ///   2. Filter stage: Apply fitler if presented. If the columns in filter are not presented, a
    ///      [`LocalTake`] will be inserted to read these columns.
    ///   3. Limit stage: Apply limit and offset if presented.
    ///   4. Take and Projection stage: Finally take the columns and project the schema.
    ///
    async fn create_plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let filter_expr = if let Some(filter) = self.filter.as_ref() {
            let planner = Planner::new(Arc::new(self.dataset.schema().into()));
            let logical_expr = planner.parse_filter(filter)?;
            Some(planner.create_physical_expr(&logical_expr)?)
        } else {
            None
        };

        // Source stage
        let mut plan = if self.nearest.is_some() {
            self.knn().await?
        } else if let Some(expr) = filter_expr {
            let columns_in_filter = column_names_in_expr(expr.as_ref());
            let filter_projection = self.dataset.schema().project(&columns_in_filter)?;
            self.scan(self.with_row_id, Arc::new(filter_projection))
        } else {
            // Full scan.
            self.scan(self.with_row_id, Arc::new(self.projections.clone()))
        };

        // Filter stage

        // Offset / limit stage

        // Take / Projection
        Ok(plan)
    }

    async fn knn(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(q) = self.nearest.as_ref() else {
            return Err(Error::IO("KNN query is not set".to_string()));
        };
        let column_id = self.dataset.schema().field_id(q.column.as_str())?;
        let use_index = self.nearest.as_ref().map(|q| q.use_index).unwrap_or(false);
        let indices = if use_index {
            self.dataset.load_indices().await?
        } else {
            vec![]
        };
        let qcol_index = indices.iter().find(|i| i.fields.contains(&column_id));

        let knn_node = if let Some(index) = qcol_index {
            // There is an index built for the column.
            // We will use the index.
            if let Some(rf) = q.refine_factor {
                if rf == 0 {
                    return Err(Error::IO("Refine factor can not be zero".to_string()));
                }
            }

            // Read from KNN index
            let knn_node = self.ann(q, &index)?; // score, _rowid
            let with_vector = self.dataset.schema().project(&[&q.column])?;
            let knn_node_with_vector = self.take(knn_node, &with_vector, false);
            if q.refine_factor.is_some() {
                self.flat_knn(knn_node_with_vector, q)?
            } else {
                knn_node_with_vector
            } // vector, score, _rowid
        } else {
            // Use flat KNN.
            let vector_scan_projection =
                Arc::new(self.dataset.schema().project(&[&q.column]).unwrap());
            let scan_node = self.scan(true, vector_scan_projection);
            self.flat_knn(scan_node, q)?
        };

        Ok(knn_node)
    }

    fn filter_knn(
        &self,
        knn_node: Arc<dyn ExecutionPlan>,
        filter_expression: Arc<dyn PhysicalExpr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let columns_in_filter = column_names_in_expr(filter_expression.as_ref());
        let columns_refs = columns_in_filter
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>();
        let filter_projection = Arc::new(self.dataset.schema().project(&columns_refs)?);

        let take_node = Arc::new(GlobalTakeExec::new(
            self.dataset.clone(),
            filter_projection.clone(),
            knn_node,
            false,
        ));
        self.filter_node(
            filter_expression,
            take_node,
            filter_projection.clone(),
            false,
        )
    }

    /// Create an Execution plan with a scan node
    fn scan(&self, with_row_id: bool, projection: Arc<Schema>) -> Arc<dyn ExecutionPlan> {
        Arc::new(LanceScanExec::new(
            self.dataset.clone(),
            self.dataset.fragments().clone(),
            projection,
            self.batch_size,
            PREFETCH_SIZE,
            with_row_id,
        ))
    }

    /// Flat KNN search.
    ///
    /// Bruteforce search for k nearest neighbors from input.
    fn flat_knn(&self, input: Arc<dyn ExecutionPlan>, q: &Query) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(KNNFlatExec::try_new(input, q.clone())?))
    }

    /// Create an Execution plan to do indexed ANN search
    fn ann(&self, q: &Query, index: &Index) -> Arc<dyn ExecutionPlan> {
        Arc::new(KNNIndexExec::new(
            self.dataset.clone(),
            &index.uuid.to_string(),
            q,
        ))
    }

    /// Take row indices produced by input plan from the dataset (with projection)
    fn take(
        &self,
        input: Arc<dyn ExecutionPlan>,
        projection: &Schema,
        drop_row_id: bool,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalTakeExec::new(
            self.dataset.clone(),
            Arc::new(projection.clone()),
            input,
            drop_row_id,
        ))
    }

    /// Global offset-limit of the result of the input plan
    fn limit_node(&self, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(GlobalLimitExec::new(
            plan,
            *self.offset.as_ref().unwrap_or(&0) as usize,
            self.limit.map(|l| l as usize),
        ))
    }

    fn filter_node(
        &self,
        filter: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
        projection: Arc<Schema>,
        drop_row_id: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let filter_node = Arc::new(FilterExec::try_new(filter, input)?);
        Ok(Arc::new(LocalTakeExec::new(
            filter_node,
            self.dataset.clone(),
            projection,
            drop_row_id,
        )))
    }
}

/// ScannerStream is a container to wrap different types of ExecNode.
#[pin_project::pin_project]
pub struct RecordBatchStream {
    #[pin]
    exec_node: SendableRecordBatchStream,
}

impl RecordBatchStream {
    pub fn new(exec_node: SendableRecordBatchStream) -> Self {
        Self { exec_node }
    }
}

impl Stream for RecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        match this.exec_node.poll_next_unpin(cx) {
            Poll::Ready(result) => {
                Poll::Ready(result.map(|r| r.map_err(|e| Error::IO(e.to_string()))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use std::path::PathBuf;

    use arrow::compute::concat_batches;
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Int32Array, Int64Array, RecordBatchReader, StringArray,
    };
    use arrow_schema::DataType;
    use futures::TryStreamExt;
    use tempfile::tempdir;

    use crate::arrow::*;
    use crate::dataset::WriteParams;

    #[tokio::test]
    async fn test_batch_size() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(
            (0..5)
                .map(|i| {
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20)),
                            Arc::new(StringArray::from_iter_values(
                                (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                            )),
                        ],
                    )
                    .unwrap()
                })
                .collect(),
        );

        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        let mut write_params = WriteParams::default();
        write_params.max_rows_per_file = 40;
        write_params.max_rows_per_group = 10;
        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut batches, test_uri, Some(write_params))
            .await
            .unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut stream = dataset
            .scan()
            .batch_size(8)
            .try_into_stream()
            .await
            .unwrap();
        for expected_len in [8, 8, 4, 8, 8, 4] {
            assert_eq!(
                stream.next().await.unwrap().unwrap().num_rows(),
                expected_len as usize
            );
        }
    }

    #[tokio::test]
    async fn test_filter_parsing() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
        ]));

        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|v| format!("s-{}", v)),
                )),
            ],
        )
        .unwrap()]);

        let mut batches: Box<dyn RecordBatchReader> = Box::new(batches);
        let test_dir = tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();
        Dataset::write(&mut batches, test_uri, None).await.unwrap();

        let dataset = Dataset::open(test_uri).await.unwrap();
        let mut scan = dataset.scan();
        assert!(scan.filter.is_none());

        scan.filter("i > 50").unwrap();
        assert_eq!(scan.filter, Some("i > 50".to_string()));

        let batches = scan
            .project(&["s"])
            .unwrap()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let batch = concat_batches(&batches[0].schema(), &batches).unwrap();

        let expected_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from_iter_values(
                (51..100).map(|v| format!("s-{}", v)),
            ))],
        )
        .unwrap();
        assert_eq!(batch, expected_batch);
    }

    #[tokio::test]
    async fn test_limit() {
        let temp = tempdir().unwrap();
        let mut file_path = PathBuf::from(temp.as_ref());
        file_path.push("limit_test.lance");
        let path = file_path.to_str().unwrap();
        let expected_batches = write_data(path).await;
        let expected_combined =
            concat_batches(&expected_batches[0].schema(), &expected_batches).unwrap();

        let dataset = Dataset::open(path).await.unwrap();
        let mut scanner = dataset.scan();
        scanner.limit(2, Some(19)).unwrap();
        let actual_batches: Vec<RecordBatch> = scanner
            .try_into_stream()
            .await
            .unwrap()
            .map(|b| b.unwrap())
            .collect::<Vec<RecordBatch>>()
            .await;
        let actual_combined = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        assert_eq!(expected_combined.slice(19, 2), actual_combined);
        // skipped 1 batch
        assert_eq!(actual_batches.len(), 2);
    }

    async fn write_data(path: &str) -> Vec<RecordBatch> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int64,
            true,
        )])) as SchemaRef;

        // Write 3 batches.
        let expected_batches: Vec<RecordBatch> = (0..3)
            .map(|batch_id| {
                let value_range = batch_id * 10..batch_id * 10 + 10;
                let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter(
                    value_range.clone().collect::<Vec<_>>(),
                ))];
                RecordBatch::try_new(schema.clone(), columns).unwrap()
            })
            .collect();
        let batches = RecordBatchBuffer::new(expected_batches.clone());
        let mut params = WriteParams::default();
        params.max_rows_per_group = 10;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, path, Some(params))
            .await
            .unwrap();
        expected_batches
    }

    async fn create_dataset() -> Arc<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, true),
            ArrowField::new("s", DataType::Utf8, true),
            ArrowField::new(
                "vec",
                DataType::List(Box::new(ArrowField::new("item", DataType::Float32, true))),
                true,
            ),
        ]));

        let vector_data = Float32Array::from_iter_values((0..3200).map(|v| v as f32));
        let vector = FixedSizeListArray::try_new(&vector_data, 32).unwrap();
        let batches = RecordBatchBuffer::new(vec![RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(StringArray::from_iter_values(
                    (0..100).map(|v| format!("s-{}", v)),
                )),
            ],
        )
        .unwrap()]);

        let mut params = WriteParams::default();
        params.max_rows_per_group = 100;
        let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
        Dataset::write(&mut reader, "memory://dataset", Some(params))
            .await
            .unwrap();
        Arc::new(Dataset::open("memory://dataset").await.unwrap())
    }

    #[tokio::test]
    async fn test_simple_scan_plan() {
        let dataset = create_dataset();
    }
}
