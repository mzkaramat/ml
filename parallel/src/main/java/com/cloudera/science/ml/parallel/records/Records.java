/**
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */
package com.cloudera.science.ml.parallel.records;

import org.apache.crunch.PCollection;

import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.parallel.summary.Summary;

/**
 *
 */
public class Records {
  private final Summary summary;
  private final Spec spec;
  private final PCollection<Record> records;
  
  public Records(PCollection<Record> records, Spec spec) {
    this.records = records;
    this.spec = spec;
    this.summary = null;
  }
  
  public Records(PCollection<Record> records, Summary summary) {
    this.records = records;
    this.spec = summary.getSpec();
    this.summary = summary;
  }

  public PCollection<Record> get() {
    return records;
  }

  public Spec getSpec() {
    return spec;
  }
  
  public boolean hasSummary() {
    return summary != null;
  }
  
  public Summary getSummary() {
    return summary;
  }
}
