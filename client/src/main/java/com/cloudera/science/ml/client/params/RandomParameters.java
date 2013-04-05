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
package com.cloudera.science.ml.client.params;

import java.util.Random;

import com.beust.jcommander.Parameter;

/**
 * Class to handle Random seed data
 *
 * <PRE>
 *  <b>--seed</b>
 *      The seed to be used for generating random number
 * </PRE>
 */
public class RandomParameters {
  @Parameter(names = "--seed",
      description = "The seed to use for the random number generator, if any")
  private Long seed;

  /**
   * Generates a Random number
   *
   * @return    {@code Random}
   */
  public synchronized Random getRandom() {
    return getRandom(0);
  }

  /**
   * Generates a Random number with specified seed value
   *
   * @param increment   value to be added to seed to generate a {@code Random} number
   * @return    {@code Random} number
   */
  public synchronized Random getRandom(long increment) {
    if (seed == null) {
      seed = System.currentTimeMillis();
    }
    return new Random(seed + increment);
  }
}
