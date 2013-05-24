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
package com.cloudera.science.ml.parallel.crossfold;

import java.util.Random;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

public class CrossfoldFn<T> extends DoFn<T, Pair<Integer, T>>{

  private int numFolds;
  private Long seed;
  
  private transient Random rand;
  
  public CrossfoldFn(int numFolds, Long seed) {
    this.numFolds = numFolds;
    this.seed = seed;
  }
  
  @Override
  public void initialize() {
    rand = (seed == null) ? new Random() : new Random(seed);
  }
  
  @Override
  public float scaleFactor() {
    return numFolds - 1.0f;
  }
  
  @Override
  public void process(T t, Emitter<Pair<Integer, T>> emitter) {
    int fold = rand.nextInt(numFolds);
    for (int i = 0; i < numFolds; i++) {
      if (i != fold) {
        emitter.emit(new Pair<Integer, T>(i, t));
      }
    }    
  }
}
