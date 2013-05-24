/**
 * Copyright (c) 2012, Cloudera, Inc. All Rights Reserved.
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

import java.util.Random;

import org.apache.crunch.MapFn;
import org.apache.crunch.Pair;

public class ShuffleFn<T> extends MapFn<T, Pair<Integer, T>> {
  private final Long seed;

  private transient Random rand;

  public ShuffleFn(Long seed) {
    this.seed = seed;
  }

  @Override
  public void initialize() {
    if (seed == null) {
      rand = new Random();
    }
  }

  @Override
  public Pair<Integer, T> map(T input) {
    if (rand == null) {
      // Salt the random generator with the hash of the first record so that
      // different runs won't all have the same sequence
      rand = new Random(seed + input.hashCode());
    }
    return Pair.of(rand.nextInt(), input);
  }
}
