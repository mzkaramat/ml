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
package com.cloudera.science.ml.classifier.simple;

import java.util.Random;

import org.junit.Test;

import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.Vectors;

/**
 *
 */
public class SimpleOnlineLearnerTest {

  private OnlineLearnerParams baseParams = OnlineLearnerParams.builder().build();
  private Random r = new Random(1729L);
  
  @Test
  public void testBasicLinear() throws Exception {
    LinRegOnlineLearner learner = new LinRegOnlineLearner(baseParams);
    for (int i = 0; i < 100000; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 3 + 2*x1 + -x2 + r.nextGaussian();
      learner.update(new LabeledVector(Vectors.of(1.0, x1, x2, x3), y));
    }
    System.out.println(learner.getClassifier());
  }
    
  @Test
  public void testBasicLogistic() throws Exception {
    LogRegOnlineLearner learner = new LogRegOnlineLearner(baseParams);
    for (int i = 0; i < 100000; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 2 + 2*x1 + -x2;
      y = (Math.exp(y)/(1.0 + Math.exp(y)) > r.nextDouble()) ? 1.0 : -1.0;
      learner.update(new LabeledVector(Vectors.of(1.0, x1, x2, x3), y));
    }
    System.out.println(learner.getClassifier());
  }
  
  @Test
  public void testBasicSVM() throws Exception {
    SVMOnlineLearner learner = new SVMOnlineLearner(baseParams);
    for (int i = 0; i < 100000; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 1.0 + 2*x1 + -4*x2;
      y = y > 0 ? 1.0 : -1.0;
      learner.update(new LabeledVector(Vectors.of(1.0, x1, x2, x3), y));
    }
    System.out.println(learner.getClassifier());
  }
}
