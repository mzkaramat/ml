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
package com.cloudera.science.ml.kmeans.parallel;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import junit.framework.Assert;

import org.apache.crunch.PCollection;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.simple.LogRegOnlineLearner;
import com.cloudera.science.ml.core.vectors.Centers;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.Vectors;
import com.cloudera.science.ml.parallel.types.MLAvros;

public class TestClusterFeaturesParallel {
  private static final double YES = 1.0;
  private static final double NO = 0.0;
  private static final Random RAND = new Random(1729l);
  private static Centers CENTERS = new Centers(
      Vectors.of(0.0, 0.0),
      Vectors.of(10.0, 10.0),
      Vectors.of(20.0, 20.0),
      Vectors.of(30.0, 30.0));
  
  private static final Collection<LabeledVector> TRAINING_VECS = makeData(5000);
  private static final PCollection<LabeledVector> TRAINING_DATA = 
      MemPipeline.typedCollectionOf(MLAvros.labeledVector(), TRAINING_VECS);
  
  private static List<LabeledVector> makeData(int pointsPerCenter) {
    List<LabeledVector> points = new ArrayList<LabeledVector>();
    for (int i = 0; i < pointsPerCenter; i++) {
      points.add(new LabeledVector(aroundCenter(0), YES));
      points.add(new LabeledVector(aroundCenter(1), NO));
      points.add(new LabeledVector(aroundCenter(2), YES));
      points.add(new LabeledVector(aroundCenter(3), NO));
    }
    Collections.shuffle(points, RAND);
    return points;
  }
  
  private static Vector aroundCenter(int centerIndex) {
    Vector center = CENTERS.get(centerIndex);
    return center.plus(Vectors.of(RAND.nextGaussian(), RAND.nextGaussian()));
  }
  
  @Test
  public void testHardClusterFeatures() {
    PCollection<LabeledVector> converted = TRAINING_DATA.parallelDo(
        new HardClusterFeaturesFn(CENTERS), MLAvros.labeledVector());
    
    Collection<LabeledVector> inputs =
        new CollectionPObject<LabeledVector>(converted).getValue();

    Assert.assertTrue(evaluate(TRAINING_VECS) < .75);
    Assert.assertTrue(evaluate(inputs) > .95);
  }
  
  @Test
  public void testTriangleClusterFeatures() {
    PCollection<LabeledVector> converted = TRAINING_DATA.parallelDo(
        new TriangleClusterFeaturesFn(CENTERS), MLAvros.labeledVector());
    
    Collection<LabeledVector> inputs =
        new CollectionPObject<LabeledVector>(converted).getValue();

    Assert.assertTrue(evaluate(TRAINING_VECS) < .75);
    Assert.assertTrue(evaluate(inputs) > .95);
  }
  
  private double evaluate(Collection<LabeledVector> inputs) {
    OnlineLearnerParams params = OnlineLearnerParams.builder().build();
    LogRegOnlineLearner learner = new LogRegOnlineLearner(params);

    for (LabeledVector input : inputs) {
      learner.update(input);
    }
    
    Classifier classifier = learner.getClassifier();
    System.out.println(classifier.getWeights());
    int successes = 0;
    for (LabeledVector input : inputs) {
      double result = classifier.apply(input.getVector());
      if (Math.abs(result - input.getLabel()) < .1) {
        successes++;
      }
    }
    double accuracy = (double) successes / inputs.size();
    System.out.println("accuracy: " + accuracy);
    return accuracy;
  }
}
