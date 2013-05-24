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
package com.cloudera.science.ml.parallel.distribute;

import java.util.Random;

import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

/**
 * Sends each input to a single random partition.
 */
public class SimpleDistributeFn<K, V> extends DistributeFn<K, V> {

  private final int numGroups;
  private final Long seed;

  private transient Random rand;

  public SimpleDistributeFn(int numGroups, Long seed) {
    this.seed = seed;
    this.numGroups = numGroups;
  }

  @Override
  public void initialize() {
    if (seed == null) {
      rand = new Random();
    }
  }
  
  @Override
  public void process(Pair<K, V> input, Emitter<Pair<Pair<K, Integer>, V>> emitter) {
    if (rand == null) {
      // Salt the random generator with the hash of the first record so that
      // different runs won't all have the same sequence
      rand = new Random(seed + input.hashCode());
    }
    emitter.emit(Pair.of(Pair.of(input.first(), rand.nextInt(numGroups)), input.second()));
  }

}
