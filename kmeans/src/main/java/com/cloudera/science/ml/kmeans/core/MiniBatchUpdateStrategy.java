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
package com.cloudera.science.ml.kmeans.core;

import java.util.List;
import java.util.Random;

import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.Weighted;
import com.cloudera.science.ml.core.vectors.WeightedSampler;
import com.google.common.collect.Lists;

/**
 *
 */
public class MiniBatchUpdateStrategy implements KMeansUpdateStrategy {

  private final int numIterations;
  private final int miniBatchSize;
  private final Random random;
  
  public MiniBatchUpdateStrategy(int numIterations, int miniBatchSize, Random random) {
    this.numIterations = numIterations;
    this.miniBatchSize = miniBatchSize;
    this.random = (random == null) ? new Random() : random;
  }
  
  @Override
  public <V extends Vector> Centers update(List<Weighted<V>> points, Centers centers) {
    int[] perCenterStepCounts = new int[centers.size()];
    WeightedSampler<V> sampler = new WeightedSampler<V>(points, random);
    for (int iter = 0; iter < numIterations; iter++) {
      // Compute closest cent for each mini-batch
      List<List<V>> centerAssignments = Lists.newArrayList();
      for (int i = 0; i < centers.size(); i++) {
        centerAssignments.add(Lists.<V>newArrayList());
      }
      for (int i = 0; i < miniBatchSize; i++) {
        V sample = sampler.sample();
        int closestId = centers.indexOfClosest(sample);
        centerAssignments.get(closestId).add(sample);
      }
      // Apply the mini-batch
      List<Vector> nextCenters = Lists.newArrayList();
      for (int i = 0; i < centerAssignments.size(); i++) {
        Vector currentCenter = centers.get(i);
        for (int j = 0; j < centerAssignments.get(i).size(); j++) {
          double eta = 1.0 / (++perCenterStepCounts[i] + 1.0);
          currentCenter = currentCenter.times(1.0 - eta);
          currentCenter = currentCenter.plus(centerAssignments.get(i).get(j).times(eta));
        }
        nextCenters.add(currentCenter);
      }
      centers = new Centers(nextCenters);
    }
    return centers;
  }
}
