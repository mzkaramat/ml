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

import junit.framework.Assert;

import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.Vectors;

/**
 *
 */
public class TestSimpleOnlineLearners {

  private OnlineLearnerParams baseParams = OnlineLearnerParams.builder().build();
  private Random r = new Random(1729L);
  
  @Test
  public void testBasicLinear() throws Exception {
    LinRegOnlineLearner learner = new LinRegOnlineLearner(baseParams);
    testSimpleOnlineLearner(learner);
  }
    
  @Test
  public void testBasicLogistic() throws Exception {
    LogRegOnlineLearner learner = new LogRegOnlineLearner(baseParams);
    testSimpleOnlineLearner(learner);
  }
  
  @Test
  public void testBasicSVM() throws Exception {
    SVMOnlineLearner learner = new SVMOnlineLearner(baseParams);
    testSimpleOnlineLearner(learner);
  }
  
  private void testSimpleOnlineLearner(SimpleOnlineLearner learner) {
    for (int i = 0; i < 100000; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 2 + 2*x1 + -x2 - 3*x3;
      y = (y > 0.0) ? 1.0 : -1.0;
      learner.update(new LabeledVector(Vectors.of(1.0, x1, x2, x3), y));
    }
    
    Classifier classifier = learner.getClassifier();
    System.out.println(classifier.getWeights());
    
    final int numTests = 100000;
    int successes = 0;
    for (int i = 0; i < numTests; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 2 + 2*x1 + -x2;
      y = (y > 0.0) ? 1.0 : 0.0;
      Vector v = Vectors.of(1.0, x1, x2, x3);
      double result = classifier.apply(v);
      if (result - y < .1) {
        successes++;
      }
    }
    double accuracy = (double) successes / numTests;
    System.out.println("accuracy: " + accuracy);
    Assert.assertTrue("accuracy " + accuracy + " < .75", accuracy > .75);
  }
}
