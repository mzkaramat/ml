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
package com.cloudera.science.ml.parallel.summary;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;


public class SummaryStats implements Serializable {

  private String name;
  private Numeric numeric;
  private Map<String, Entry> histogram;
  private Double scale;
  
  // For serialization
  private SummaryStats() { }
  
  public static SummaryStats createNumeric(String name, Numeric numeric) {
    return new SummaryStats(name, numeric);
  }
  
  public static SummaryStats createCategorical(String name, List<String> levels) {
    Map<String, Entry> histogram = Maps.newHashMap();
    for (int i = 0; i < levels.size(); i++) {
      histogram.put(levels.get(i), new Entry(i));
    }
    return new SummaryStats(name, histogram);
  }
  
  SummaryStats(String name) {
    this.name = name;
    this.numeric = null;
    this.histogram = null;
  }
  
  SummaryStats(String name, Numeric numeric) {
    this.name = name;
    this.numeric = Preconditions.checkNotNull(numeric);
    this.histogram = null;
  }
  
  SummaryStats(String name, Map<String, Entry> histogram) {
    this.name = name;
    this.numeric = null;
    this.histogram = Preconditions.checkNotNull(histogram);
  }
  
  public boolean isEmpty() {
    return numeric == null && histogram == null;
  }
  
  public boolean isNumeric() {
    return numeric != null;
  }
  
  public String getName() {
    return name;
  }

  public double getScale() {
    return scale == null ? 1.0 : scale;
  }
  
  public double mean() {
    return numeric == null ? Double.NaN : numeric.mean();
  }
  
  public double stdDev() {
    return numeric == null ? Double.NaN : numeric.stdDev();
  }
  
  public double range() {
    return numeric == null ? Double.NaN : numeric.range();
  }
  
  public double min() {
    return numeric == null ? Double.NaN : numeric.min();
  }
  
  public double max() {
    return numeric == null ? Double.NaN : numeric.max();
  }
  
  public long getMissing() {
    return numeric == null ? 0L : numeric.getMissing();
  }
  
  public String getTransform() {
    return numeric == null ? null : numeric.getTransform();
  }
  
  public List<String> getLevels() {
    if (histogram == null) {
      return ImmutableList.of();
    }
    List<String> levels = Lists.newArrayList(histogram.keySet());
    Collections.sort(levels);
    return levels;
  }
  
  public int numLevels() {
    return histogram == null ? 1 : histogram.size();
  }
  
  public int index(String value) {
    if (histogram == null) {
      return -1;
    }
    Entry e = histogram.get(value);
    if (e == null) {
      return -1;
    } else {
      return e.id;
    }
  }  
}