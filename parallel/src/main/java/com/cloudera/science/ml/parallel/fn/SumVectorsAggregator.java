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
package com.cloudera.science.ml.parallel.fn;

import org.apache.crunch.Aggregator;
import org.apache.crunch.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;

import com.google.common.collect.ImmutableList;

/**
 * An aggregator that handles the common operation of summing up a large number
 * of {@code Vector} objects and their counts so that they may be averaged.
 */
public class SumVectorsAggregator<V extends Vector> implements Aggregator<Pair<V, Long>> {

  private transient Vector sum;
  private long count;
  
  @Override
  public void initialize(Configuration conf) {
    reset();
  }

  @Override
  public void reset() {
    sum = null;
    count = 0L;
  }

  @Override
  public Iterable<Pair<V, Long>> results() {
    return ImmutableList.of(Pair.of((V) sum, count));
  }

  @Override
  public void update(Pair<V, Long> in) {
    if (sum == null) {
      sum = in.first().clone();
    } else {
      sum = sum.plus(in.first());
    }
    count += in.second();
  }
}
