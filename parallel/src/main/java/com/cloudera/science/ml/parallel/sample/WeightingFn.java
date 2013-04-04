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
package com.cloudera.science.ml.parallel.sample;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.core.records.Specs;

/**
 * Create a weighted {@code Record} based on a specified field from the record {@code Spec}.
 */
public class WeightingFn extends DoFn<Record, Pair<Record, Double>> {

  private final int columnId;
  private final boolean invert;
  
  public WeightingFn(Spec spec, String weightField, boolean invert) {
    if (weightField != null) {
      this.columnId = Specs.getFieldId(spec, weightField);
    } else {
      this.columnId = -1;
    }
    this.invert = invert;
  }

  @Override
  public void process(Record rec, Emitter<Pair<Record, Double>> emitter) {
    if (columnId < 0) {
      emitter.emit(Pair.of(rec, 1.0));
    } else {
      double w = rec.getAsDouble(columnId);
      if (!Double.isNaN(w) && w != 0.0) {
        emitter.emit(Pair.of(rec, invert ? 1.0 / w : w));
      }
    }
  }
}
