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

import java.util.List;

import org.apache.crunch.MapFn;
import org.apache.crunch.Pair;

import com.cloudera.science.ml.core.records.Record;

/**
 * Given a weighted {@code Record}, extract a set of columns to use as a grouping key.
 */
public class RecordGroupFn extends MapFn<Pair<Record, Double>, String> {

  private final List<Integer> columnIds;
  
  public RecordGroupFn(List<Integer> columnIds) {
    this.columnIds = columnIds;
  }
  
  @Override
  public String map(Pair<Record, Double> in) {
    StringBuilder sb = new StringBuilder();
    for (Integer columnId : columnIds) {
      sb.append(in.first().getAsString(columnId));
    }
    return sb.toString();
  }
}

