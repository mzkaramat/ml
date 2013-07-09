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

import java.util.ArrayList;
import java.util.Random;

import junit.framework.Assert;

import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.rank.LogRegRankOnlineLearner;
import com.cloudera.science.ml.classifier.rank.RankOnlineLearner;
import com.cloudera.science.ml.classifier.rank.SVMRankOnlineLearner;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.Vectors;

public class TestRankOnlineLearners {
  private OnlineLearnerParams baseParams = OnlineLearnerParams.builder().build();
  private Random r = new Random(1729L);
  
  @Test
  public void testBasicLogistic() throws Exception {
    LogRegRankOnlineLearner learner = new LogRegRankOnlineLearner(baseParams);
    testRankOnlineLearner(learner);
  }
  
  @Test
  public void testBasicSVM() throws Exception {
    SVMRankOnlineLearner learner = new SVMRankOnlineLearner(baseParams);
    testRankOnlineLearner(learner);
  }
  
  private void testRankOnlineLearner(RankOnlineLearner learner) {
    ArrayList<LabeledVector> positive = new ArrayList<LabeledVector>();
    ArrayList<LabeledVector> negative = new ArrayList<LabeledVector>();
    for (int i = 0; i < 100000; i++) {
      double x1 = r.nextGaussian();
      double x2 = r.nextGaussian();
      double x3 = r.nextGaussian();
      double y = 2 + 2*x1 + -x2 - 3*x3;
      y = (y > 0.0) ? 1.0 : -1.0;
      LabeledVector vec = new LabeledVector(Vectors.of(1.0, x1, x2, x3), y);
      if (y == 1.0) {
        positive.add(vec);
      } else {
        negative.add(vec);
      }
    }
    for (int i = 0; i < 100000; i++) {
      LabeledVector posVec = positive.get(r.nextInt(positive.size()));
      LabeledVector negVec = negative.get(r.nextInt(negative.size()));
      learner.update(posVec, negVec);
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
