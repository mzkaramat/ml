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
package com.cloudera.science.ml.core.summary;

import java.io.Serializable;
import java.util.List;
import java.util.Set;

import com.cloudera.science.ml.core.records.DataType;
import com.cloudera.science.ml.core.records.RecordSpec;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

public class Summary implements Serializable {

  private List<SummaryStats> stats = Lists.newArrayList();
  private long recordCount;
  private int fieldCount;
  
  public Summary() {}
  
  public Summary(long recordCount, int fieldCount, List<SummaryStats> stats) {
    this.recordCount = recordCount;
    this.fieldCount = fieldCount;
    this.stats = stats;
  }
  
  public RecordSpec getSpec() {
    RecordSpec.Builder rsb = RecordSpec.builder();
    for (int i = 0; i < stats.size(); i++) {
      SummaryStats ss = stats.get(i);
      if (ss != null) {
        String field = ss.getName();
        if (field == null) {
          field = "c" + i;
        }
        if (ss.isNumeric()) {
          rsb.add(field, DataType.DOUBLE);
        } else {
          rsb.add(field, DataType.STRING);
        }
      } else {
        rsb.add("c" + i, DataType.STRING);
      }
    }
    return rsb.build();
  }
  
  public List<SummaryStats> getAllStats() {
    return stats;
  }
  
  public long getRecordCount() {
    return recordCount;
  }
  
  public int getFieldCount() {
    return fieldCount;
  }
  
  public Set<Integer> getIgnoredColumns() {
    if (stats.isEmpty()) {
      return ImmutableSet.of();
    }
    Set<Integer> ignored = Sets.newHashSet();
    for (int i = 0; i < stats.size(); i++) {
      SummaryStats ss = stats.get(i);
      if (ss == null || ss.isEmpty()) {
        ignored.add(i);
      }
    }
    return ignored;
  }
  
  public SummaryStats getStats(int field) {
    if (field >= stats.size()) {
      return null;
    }
    return stats.get(field);
  }
  
  public int getNetLevels() {
    int netLevels = 0;
    for (SummaryStats ss : stats) {
      if (ss != null && !ss.isEmpty()) {
        netLevels += ss.numLevels() - 1;
      }
    }
    return netLevels;
  }
}
