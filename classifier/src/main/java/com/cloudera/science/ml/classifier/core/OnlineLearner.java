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
package com.cloudera.science.ml.classifier.core;

import java.io.Serializable;

import com.google.common.base.Preconditions;


/**
 *
 */
public interface OnlineLearner extends Serializable {
  
  Classifier getClassifier();
  
  public static class Params implements Serializable {

      private final int numFeatures;
      private final double lambda;
      private final EtaUpdate etaUpdate;
      private final boolean pegasos;
      private final double l1Radius;
      private final int l1Iterations;
      
      private Params(int numFeatures, double lambda, EtaUpdate etaUpdate,
          boolean pegasos, double l1Radius, int l1Iterations) {
        this.numFeatures = numFeatures;
        this.lambda = lambda;
        this.etaUpdate = etaUpdate;
        this.pegasos = pegasos;
        this.l1Radius = l1Radius;
        this.l1Iterations = l1Iterations;
      }

      public static Builder builder(int numFeatures) {
        return new Builder(numFeatures);
      }

      public WeightVector createWeights() {
        return new WeightVector(numFeatures);
      }

      public double lambda() {
        return lambda;
      }
      
      public double eta(int iteration) {
        return etaUpdate.compute(lambda, iteration);
      }
      
      public void updateWeights(WeightVector weights, int iteration) {
        if (pegasos) {
          weights.pegasosProjection(lambda);
        }
        if (l1Iterations > 0 && iteration % l1Iterations == 0) {
          weights.approxRegularizeL1(l1Radius);
        }
      }
      
      public static class Builder {
        private final int numFeatures;
        private double lambda = 0.0;
        private EtaUpdate etaUpdate = EtaUpdate.BASIC_ETA;
        private boolean pegasos = false;
        private int l1Iterations = -1;
        private double l1Radius = 10.0;
        
        public Builder(int numFeatures) {
          Preconditions.checkArgument(numFeatures > 0);
          this.numFeatures = numFeatures;
        }
        
        public Builder L2(double lambda) {
          Preconditions.checkArgument(lambda >= 0.0);
          this.lambda = lambda;
          return this;
        }
        
        public Builder etaUpdate(EtaUpdate etaUpdate) {
          this.etaUpdate = Preconditions.checkNotNull(etaUpdate);
          return this;
        }
        
        public Builder pegasos(boolean pegasos) {
          this.pegasos = pegasos;
          return this;
        }
        
        public Builder L1(double radius, int iterations) {
          Preconditions.checkArgument(iterations > 0, "L1 iterations must be > 0");
          Preconditions.checkArgument(radius > 0.0, "L1 radius must be > 0");
          this.l1Radius = radius;
          this.l1Iterations = iterations;
          return this;
        }
        
        public Params build() {
          // Do some checking for problematic interactions
          if (lambda == 0.0) {
            if (etaUpdate == EtaUpdate.PEGASOS_ETA) {
              throw new IllegalStateException("PEGASOS_ETA requires L2 lambda > 0");
            }
            if (pegasos) {
              throw new IllegalStateException("Pegasos projection requires L2 lambda > 0");
            }
          }
          return new Params(numFeatures, lambda, etaUpdate, pegasos, l1Radius, l1Iterations);
        }
      }
  }
}
