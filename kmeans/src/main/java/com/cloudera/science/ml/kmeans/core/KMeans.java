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
package com.cloudera.science.ml.kmeans.core;

import java.util.List;
import java.util.Random;

import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.Weighted;
import com.google.common.base.Preconditions;

/**
 * An in-memory implementation of the k-means algorithm
 * that can be configured to create various numbers of clusters using different
 * {@link KMeansInitStrategy} initialization strategies and updated with different
 * {@link KMeansUpdateStrategy} rules. For more details on k-means and
 * its properties, please see <a href="http://en.wikipedia.org/wiki/K-means_clustering">the
 * Wikipedia page.</a>
 */
public class KMeans {

  private static final Logger LOG = LoggerFactory.getLogger(KMeans.class);
  
  private final KMeansInitStrategy initStrategy;
  private final KMeansUpdateStrategy updateStrategy;
  
  /**
   * Constructor that uses the k-means++ initialization strategy and
   * 100 iterations of Lloyd's algorithm.
   */
  public KMeans() {
    this(KMeansInitStrategy.PLUS_PLUS, new LloydsUpdateStrategy(100));
  }
  
  /**
   * Creates an in-memory k-means execution engine.
   * 
   * @param initStrategy The initialization strategy for the k-means algorithm
   * @param updateStrategy The update strategy for the k-means algorithm
   */
  public KMeans(
      KMeansInitStrategy initStrategy,
      KMeansUpdateStrategy updateStrategy) {
    this.initStrategy = Preconditions.checkNotNull(initStrategy);
    this.updateStrategy = Preconditions.checkNotNull(updateStrategy);
  }
  
  /**
   * Apply the configured k-means initialization strategy followed by
   * the k-means update strategy for the given list of points to yield the given number
   * of clusters.
   * 
   * @param points The weighted points to cluster
   * @param numClusters Number of clusters to create
   * @return The {@code Centers} created from the computations
   */
  public <V extends Vector> Centers compute(List<Weighted<V>> points, int numClusters) {
    return compute(points, numClusters, null);
  }

  /**
   * Apply the configured k-means initialization strategy followed by
   * the k-means update strategy for the given list of points to yield the given number
   * of clusters.
   * 
   * @param points The weighted points to cluster
   * @param numClusters Number of clusters to create
   * @param random The random number generator to use
   * @return The {@code Centers} created from the computations
   */
  public <V extends Vector> Centers compute(List<Weighted<V>> points, int numClusters, Random random) {
    Preconditions.checkArgument(numClusters > 0);
    Centers initial = initStrategy.apply(points, numClusters, random);
    Centers updated = updateStrategy.update(points, initial);
    if (initial.size() != updated.size()) {
      LOG.warn(String.format(
          "Centers collapsed: client requested %d centers, but only %d were found",
          initial.size(), updated.size()));
    }
    return updated;
  }
}
