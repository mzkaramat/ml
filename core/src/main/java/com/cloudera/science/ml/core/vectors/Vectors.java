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
package com.cloudera.science.ml.core.vectors;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

/**
 * Factory methods for working with {@code Vector} objects.
 */
public final class Vectors {

  /**
   * Converts the given {@code Vector} into a {@code double[]}.
   * 
   * @param v The vector to convert
   * @return The resulting array of values
   */
  public static double[] toArray(Vector v) {
    double[] ret = new double[v.size()];
    for (int i = 0; i < ret.length; i++) {
      ret[i] = v.getQuick(i);
    }
    return ret;
  }
  
  /**
   * Creates a dense {@code Vector} instance from the given values.
   * 
   * @param values The array of values to turn into a {@code Vector}
   * @return The resulting {@code Vector}
   */
  public static Vector of(double... values) {
    return new DenseVector(values);
  }
  
  /**
   * Constructs a {@code DenseVector} of the given size.
   * 
   * @param size The size of the dense vector to create
   * @return The new {@code DenseVector}
   */
  public static Vector dense(int size) {
    double[] d = new double[size];
    return new DenseVector(d);
  }
  
  /**
   * Constructs a {@code RandomAccessSparseVector} of the given size.
   * 
   * @param size The size of the sparse vector to create
   * @return The new {@code RandomAccessSparseVector}
   */
  public static Vector sparse(int size) {
    return new RandomAccessSparseVector(size);
  }

  /**
   * Constructs a {@code NamedVector} from the given name and
   * values.
   * 
   * @param name The name of the vector
   * @param v The values it contains
   * @return A new {@code NamedVector}
   */
  public static Vector named(String name, double... v) {
    return new NamedVector(of(v), name);
  }
  
  // Not instantiated
  private Vectors() {}
}
