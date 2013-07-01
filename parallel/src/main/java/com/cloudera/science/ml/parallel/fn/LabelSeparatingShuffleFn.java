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

import java.util.Random;

import org.apache.crunch.Pair;

import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 * A function that both shuffles inputs and groups them by binary label.
 */
public class LabelSeparatingShuffleFn extends ShuffleFn<LabeledVector> {

  private double rarerLabel;
  
  public LabelSeparatingShuffleFn(Long seed, double rarerLabel) {
    super(seed);
    this.rarerLabel = rarerLabel;
  }
  
  @Override
  public Pair<Integer, LabeledVector> map(LabeledVector input) {
    if (rand == null) {
      // Salt the random generator with the hash of the first record so that
      // parallel runs won't all have the same sequence
      rand = new Random(seed + input.hashCode());
    }
    // rarerLabel gets negatives so it'll come first
    return Pair.of((rand.nextInt(Integer.MAX_VALUE - 2) + 1)
        * input.getLabel() == rarerLabel ? -1 : 1, input);
  }
}
