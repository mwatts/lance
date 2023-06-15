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

use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchReader};
use arrow_schema::{DataType, Field, FieldRef, Schema};
use criterion::{criterion_group, criterion_main, Criterion};
use lance::{
    arrow::RecordBatchBuffer,
    dataset::{WriteMode, WriteParams},
    utils::testing::generate_random_array,
    Dataset,
};
use pprof::criterion::{Output, PProfProfiler};

fn bench_diskann(c: &mut Criterion) {}

async fn create_dataset(path: &std::path::Path, mode: WriteMode) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            FieldRef::new(Field::new("item", DataType::Float32, true)),
            128,
        ),
        false,
    )]));

    let num_rows = 100_000;
    let batch_size = 10000;
    let dim = 768;
    let batches = RecordBatchBuffer::new(
        (0..(num_rows / batch_size) as i32)
            .map(|_| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(
                        FixedSizeListArray::try_new(generate_random_array(num_rows * dim), dim)
                            .unwrap(),
                    )],
                )
                .unwrap()
            })
            .collect(),
    );

    let test_uri = path.to_str().unwrap();
    std::fs::remove_dir_all(test_uri).map_or_else(|_| println!("{} not exists", test_uri), |_| {});
    let mut write_params = WriteParams::default();
    write_params.max_rows_per_file = num_rows as usize;
    write_params.max_rows_per_group = batch_size as usize;
    write_params.mode = mode;

    let mut reader: Box<dyn RecordBatchReader> = Box::new(batches);
    let dataset = Dataset::write(&mut reader, test_uri, Some(write_params))
        .await
        .unwrap();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_diskann);

criterion_main!(benches);
