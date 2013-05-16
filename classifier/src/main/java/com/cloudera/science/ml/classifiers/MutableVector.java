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

package com.cloudera.science.ml.classifiers;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.core.vectors.LabeledVector;

public class MutableVector {

  private float[] weights_;
  private double scale_;
  private double squared_norm_;
  private int dimensions_;

  public MutableVector(int dimensionality) {
    scale_ = 1.0;
    dimensions_ = dimensionality;
    if (dimensions_ <= 0) {
      System.out.println("Illegal dimensionality of weight vector less than 1.");
      System.out.println("dimensions_: " + dimensions_);
      System.exit(1);
    }

    weights_ = new float[dimensions_];
    for (int i = 0; i < dimensions_; ++i) {
      weights_[i] = 0;
    }
  }

  public MutableVector(MutableVector weight_vector) {
    scale_ = weight_vector.scale_;
    squared_norm_ = weight_vector.squared_norm_;
    dimensions_ = weight_vector.dimensions_;

    weights_ = new float[dimensions_];

    for (int i = 0; i < dimensions_; ++i) {
      weights_[i] = weight_vector.weights_[i];
    }
  }

  public float innerProduct(LabeledVector x, float x_scale) {
    float inner_product = 0.0f;
    for (Vector.Element element : x.getVector()) {
      inner_product += weights_[element.index()] * element.get();
    }
    inner_product *= x_scale;
    inner_product *= scale_;
    return inner_product;
  }
  
  public float innerProduct(LabeledVector x) {
    return innerProduct(x, 1.0f);
  }

  public float innerProductOnDifference(LabeledVector a,
      LabeledVector b, float x_scale) {
    //   <x_scale * (a - b), w>
    // = <x_scale * a - x_scale * b, w>
    // = <x_scale * a, w> + <-1.0 * x_scale * b, w>

    float inner_product = 0.0f;
    inner_product += innerProduct(a, x_scale);
    inner_product += innerProduct(b, -1.0f * x_scale);
    return inner_product;
  }
  
  public float innerProductOnDifference(LabeledVector a, LabeledVector b) {
    return innerProductOnDifference(a, b, 1.0f);
  }

  void addVector(LabeledVector x, float x_scale) {
//    if (x.FeatureAt(x.NumFeatures() - 1) > dimensions_) {
//      throw new IllegalArgumentException("Feature " + x.FeatureAt(x.NumFeatures() - 1) 
//          + " exceeds dimensionality of weight vector: " 
//          + dimensions_);
//    }

    float inner_product = 0.0f;
    for (Vector.Element element : x.getVector()) {
      float this_x_value = (float)element.get() * x_scale;
      int this_x_feature = element.index();
      inner_product += weights_[this_x_feature] * this_x_value;
      weights_[this_x_feature] += this_x_value / scale_;
    }
    squared_norm_ += x.getSquaredNorm() * x_scale * x_scale +
        (2.0 * scale_ * inner_product); 
  }

  void scaleBy(double scaling_factor) {
    // Take care of any numerical difficulties.
    if (scale_ < 0.00000000001) ScaleToOne();

    // Do the scaling.
    squared_norm_ *= (scaling_factor * scaling_factor);
    if (scaling_factor > 0.0) {
      scale_ *= scaling_factor;
    } else {
      throw new IllegalStateException(
          "Error: scaling weight vector by non-positive value!\n " 
              + "This can cause numerical errors in PEGASOS projection.\n "
              + "This is likely due to too large a value of eta * lambda.\n ");
    }
  }

  float ValueOf(int index) {
    if (index < 0) {
      System.out.println("Illegal index " + index + " in ValueOf. ");
      System.exit(1);
    }
    if (index >= dimensions_) {
      return 0;
    }
    return weights_[index] * (float)scale_;
  }

  void ProjectToL1Ball(float lambda, float epsilon) {
    // Re-scale lambda.
    lambda = lambda / (float)scale_;

    // Bail out early if possible.
    float current_l1 = 0.0f;
    float max_value = 0.0f;
    List<Float> non_zeros = new ArrayList<Float>();
    for (int i = 0; i < dimensions_; ++i) {
      if (weights_[i] != 0.0) {
        non_zeros.add(Math.abs(weights_[i]));
      } else {
        continue;
      }
      current_l1 += Math.abs(weights_[i]);
      if (Math.abs(weights_[i]) > max_value) {
        max_value = Math.abs(weights_[i]);
      }
    }
    if (current_l1 <= (1.0 + epsilon) * lambda) return;

    float min = 0;
    float max = max_value;
    float theta = 0.0f;
    while (current_l1 >  (1.0 + epsilon) * lambda ||
        current_l1 < lambda) {
      theta = (max + min) / 2.0f;
      current_l1 = 0.0f;
      for (int i = 0; i < non_zeros.size(); ++i) {
        current_l1 += Math.max(0, non_zeros.get(i) - theta);
      }
      if (current_l1 <= lambda) {
        max = theta;
      } else {
        min = theta;
      }
    }

    for (int i = 0; i < dimensions_; ++i) {
      if (weights_[i] > 0) weights_[i] = Math.max(0, weights_[i] - theta);
      if (weights_[i] < 0) weights_[i] = Math.min(0, weights_[i] + theta);
    } 
  }

/*
  void ProjectToL1Ball(float lambda) {
    // Bail out early if possible.
    float current_l1 = 0.0f;
    for (int i = 0; i < dimensions_; ++i) {
      if (Math.abs(ValueOf(i)) > 0) current_l1 += Math.abs(ValueOf(i));
    }
    if (current_l1 < lambda) return;

    List<Integer> workspace_a;
    List<Integer> workspace_b;
    List<Integer> workspace_c;
    List<Integer> U = &workspace_a;
    List<Integer> L = &workspace_b;
    List<Integer> G = &workspace_c;
    List<Integer> temp;
    // Populate U with all non-zero elements in weight vector.
    for (int i = 0; i < dimensions_; ++i) {
      if (Math.abs(ValueOf(i)) > 0) {
        U.add(i);
        current_l1 += Math.abs(ValueOf(i));
      }
    }

    // Find the value of theta.
    double partial_pivot = 0;
    double partial_sum = 0;
    while (U.size() > 0) {
      G.clear();
      L.clear();
      int k = (*U)[(int)(rand() % U.size())];
      float pivot_k = Math.abs(ValueOf(k));
      float partial_sum_delta = Math.abs(ValueOf(k));
      float partial_pivot_delta = 1.0f;
      // Partition U using pivot_k.
      for (int i = 0; i < U.size(); ++i) {
        float w_i = Math.abs(ValueOf(U.get(i)));
        if (w_i >= pivot_k) {
          if (U.get(i) != k) {
            partial_sum_delta += w_i;
            partial_pivot_delta += 1.0;
            G.add(U.get(i));
          }
        } else {
          L.add(U.get(i));
        }
      }
      if ((partial_sum + partial_sum_delta) -
          pivot_k * (partial_pivot + partial_pivot_delta) < lambda) {
        partial_sum += partial_sum_delta;
        partial_pivot += partial_pivot_delta;
        temp = U;
        U = L;
        L = temp;
      } else {
        temp = U;
        U = G;
        G = temp;
      }
    }

    // Perform the projection.
    float theta = ((float)partial_sum - (float)lambda) / partial_pivot;  
    squared_norm_ = 0.0;
    for (int i = 0; i < dimensions_; ++i) {
      if (ValueOf(i) == 0.0) continue;
      int sign = (ValueOf(i) > 0) ? 1 : -1;
      weights_[i] = sign * Math.max((sign * ValueOf(i) - theta), 0); 
      squared_norm_ += weights_[i] * weights_[i];
    }
    scale_ = 1.0;
  }
*/

  void ScaleToOne() {
    for (int i = 0; i < dimensions_; ++i) {
      weights_[i] *= scale_;
    }
    scale_ = 1.0;
  }
  
  public float getSquaredNorm() { return (float)squared_norm_; }
  public int getDimensions() { return dimensions_; }
}
