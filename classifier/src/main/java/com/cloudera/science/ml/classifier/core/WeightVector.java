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

package com.cloudera.science.ml.classifier.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.vectors.Vectors;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;

public class WeightVector implements Serializable {

  private static final double MIN_SCALING_FACTOR = 0.0000001;
  
  private float[] weights;
  private double scale;
  private double squaredNorm;
  private int dimensions;

  public WeightVector(int dimensionality) {
    Preconditions.checkArgument(dimensionality > 0, "Illegal dimensionality of weight vector: %d",
        dimensionality);
    scale = 1.0;
    dimensions = dimensionality;
    weights = new float[dimensions];
    for (int i = 0; i < dimensions; ++i) {
      weights[i] = 0;
    }
  }

  public WeightVector(double... weights) {
    this(weights.length);
    this.addVector(Vectors.of(weights), 1.0);
  }
  
  public WeightVector(WeightVector other) {
    scale = other.scale;
    squaredNorm = other.squaredNorm;
    dimensions = other.dimensions;

    weights = new float[dimensions];
    System.arraycopy(other.weights, 0, weights, 0, dimensions);
  }

  public double innerProduct(Vector vec, double vecScale) {
    double innerProduct = 0.0;
    Iterator<Vector.Element> iter = vec.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      innerProduct += weights[element.index()] * element.get();
    }
    innerProduct *= vecScale;
    innerProduct *= scale;
    return innerProduct;
  }
  
  public double innerProduct(Vector x) {
    return innerProduct(x, 1.0);
  }

  public double innerProductOnDifference(Vector a,
      Vector b, double vecScale) {
    //   <vecScale * (a - b), w>
    // = <vecScale * a - vecScale * b, w>
    // = <vecScale * a, w> + <-1.0 * vecScale * b, w>
    double innerProduct = 0.0;
    innerProduct += innerProduct(a, vecScale);
    innerProduct += innerProduct(b, -1.0 * vecScale);
    return innerProduct;
  }
  
  public double innerProductOnDifference(Vector a, Vector b) {
    return innerProductOnDifference(a, b, 1.0);
  }

  public void addVector(Vector vec, double vecScale) {
    double innerProduct = 0.0;
    Iterator<Vector.Element> iter = vec.iterateNonZero(); 
    while (iter.hasNext()) {
      Vector.Element element = iter.next();
      double value = element.get() * vecScale;
      int index = element.index();
      innerProduct += weights[index] * value;
      weights[index] += value / scale;
    }
    squaredNorm += vec.norm(2) * vecScale * vecScale +
        (2.0 * scale * innerProduct); 
  }

  public void scaleBy(double scalingFactor) {
    // Take care of any numerical difficulties.
    if (scale < 0.00000000001) scaleToOne();

    // Do the scaling.
    squaredNorm *= (scalingFactor * scalingFactor);
    if (scalingFactor > 0.0) {
      scale *= scalingFactor;
    } else {
      throw new IllegalStateException(
          "Error: scaling weight vector by non-positive value!\n " 
              + "This can cause numerical errors in PEGASOS projection.\n "
              + "This is likely due to too large a value of eta * lambda.\n ");
    }
  }

  public double get(int index) {
    if (index >= dimensions) {
      return 0;
    }
    return weights[index] * scale;
  }
  
  public void regularizeL2(double eta, double lambda) {
    double scalingFactor = 1.0 - (eta * lambda);
    if (scalingFactor > MIN_SCALING_FACTOR) {
      scaleBy(scalingFactor);  
    } else {
      scaleBy(MIN_SCALING_FACTOR); 
    }
  }
  
  public void pegasosProjection(double lambda) {
    double projectionVal = 1.0 / Math.sqrt(lambda * getSquaredNorm());
    if (projectionVal < 1.0) {
      scaleBy(projectionVal);
    }
  }
  
  public void approxRegularizeL1(double lambda) {
    // Re-scale lambda
    lambda = lambda / scale;
    
    double currentL1 = 0.0;
    int currentL0 = 0;
    for (int i = 0; i < dimensions; i++) {
      if (weights[i] != 0.0f) {
        currentL1 += Math.abs(weights[i]);
        currentL0++;
      }
    }
    
    if (currentL0 > 0) {
      float tau = (float) (Math.max(currentL1 - lambda, 0.0) / currentL0);
      if (tau > 0.0f) {
        squaredNorm = 0.0;
        for (int i = 0; i < dimensions; i++) {
          if (weights[i] > 0) weights[i] = Math.max(0, weights[i] - tau);
          if (weights[i] < 0) weights[i] = Math.min(0, weights[i] + tau);
          squaredNorm += weights[i] * weights[i] * scale * scale;
        }
      }
    }
  }
  
  public void regularizeL1(double lambda, double epsilon) {
    // Re-scale lambda.
    lambda = lambda / scale;

    // Bail out early if possible.
    double currentL1 = 0.0;
    float maxValue = 0.0f;
    List<Float> nonZeros = new ArrayList<Float>();
    for (int i = 0; i < dimensions; ++i) {
      if (weights[i] != 0.0f) {
        nonZeros.add(Math.abs(weights[i]));
        currentL1 += Math.abs(weights[i]);
        if (Math.abs(weights[i]) > maxValue) {
          maxValue = Math.abs(weights[i]);
        }
      }
    }
    if (currentL1 <= (1.0 + epsilon) * lambda) return;

    float min = 0;
    float max = maxValue;
    float theta = 0.0f;
    while (currentL1 > (1.0 + epsilon) * lambda ||
        currentL1 < lambda) {
      theta = (max + min) / 2.0f;
      currentL1 = 0.0f;
      for (int i = 0; i < nonZeros.size(); ++i) {
        currentL1 += Math.max(0, nonZeros.get(i) - theta);
      }
      if (currentL1 <= lambda) {
        max = theta;
      } else {
        min = theta;
      }
    }

    squaredNorm = 0.0;
    for (int i = 0; i < dimensions; ++i) {
      if (weights[i] > 0) weights[i] = Math.max(0, weights[i] - theta);
      if (weights[i] < 0) weights[i] = Math.min(0, weights[i] + theta);
      squaredNorm += weights[i] * weights[i];
    } 
  }

  private void scaleToOne() {
    for (int i = 0; i < dimensions; ++i) {
      weights[i] *= scale;
    }
    scale = 1.0;
  }
  
  public double getSquaredNorm() { return squaredNorm; }
  public int getDimensions() { return dimensions; }
  
  @Override
  public int hashCode() {
    return Arrays.hashCode(weights) + Doubles.hashCode(scale);
  }
  
  @Override
  public boolean equals(Object other) {
    if (other == null || !(other instanceof WeightVector)) {
      return false;
    }
    WeightVector wv = (WeightVector) other;
    if (dimensions != wv.dimensions || squaredNorm != wv.squaredNorm ||
        scale != wv.scale) {
      return false;
    }
    return Arrays.equals(weights, wv.weights);
  }
  
  @Override
  public String toString() {
    scaleToOne();
    return Arrays.toString(weights);
  }
}
