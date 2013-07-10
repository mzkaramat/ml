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

package com.cloudera.science.ml.classifier.parallel;

import java.util.Arrays;
import java.util.Collection;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.impl.mem.MemPipeline;
import org.apache.crunch.materialize.pobject.CollectionPObject;
import org.junit.Assert;
import org.junit.Test;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.OnlineLearnerParams;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.core.vectors.LabeledVectors;
import com.cloudera.science.ml.parallel.crossfold.CrossfoldFn;
import com.cloudera.science.ml.parallel.distribute.SimpleDistributeFn;
import com.cloudera.science.ml.parallel.fn.LabelSeparatingShuffleFn;
import com.cloudera.science.ml.parallel.fn.ShuffleFn;
import com.cloudera.science.ml.parallel.types.MLAvros;

public class TestBalancedLearning {
  private static final long seed = 1729L;
  private static final double YES = 1.0;
  private static final double NO = 0.0;
  private static final PCollection<LabeledVector> TRAINING_DATA = 
      MemPipeline.typedCollectionOf(
          MLAvros.labeledVector(),
          LabeledVectors.of(NO, -5.0, -10.0),
          LabeledVectors.of(YES, -10.0, -5.0),
          LabeledVectors.of(NO, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0),
          LabeledVectors.of(YES, -5.0, -10.0),
          LabeledVectors.of(YES, -10.0, -5.0),
          LabeledVectors.of(YES, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0),
          LabeledVectors.of(NO, -5.0, -10.0),
          LabeledVectors.of(YES, -10.0, -5.0),
          LabeledVectors.of(YES, 5.0, 10.0),
          LabeledVectors.of(YES, 10.0, 5.0));
  
  @Test
  public void testBalancedLearning() {
    CountingLearner learner = new CountingLearner();
    BalancedFitFn fitFn = new BalancedFitFn(Arrays.asList((SimpleOnlineLearner)learner));
    ShuffleFn<LabeledVector> shuffleFn = new LabelSeparatingShuffleFn(seed, NO);
    ParallelLearner parallelLearner = new ParallelLearner();
    PCollection<OnlineLearnerRun> pruns = parallelLearner.runPipeline(TRAINING_DATA,
        shuffleFn, new CrossfoldFn<Pair<Integer, LabeledVector>>(1, seed),
        new SimpleDistributeFn<Integer, Pair<Integer, LabeledVector>>(1, seed),
        fitFn);
    
    Collection<OnlineLearnerRun> runs =
        new CollectionPObject<OnlineLearnerRun>(pruns).getValue();
    Assert.assertEquals(1, runs.size());
    Assert.assertEquals(9, learner.numPositive);
    Assert.assertEquals(learner.numNegative, learner.numPositive);
  }
  
  @Test (expected = IllegalStateException.class)
  public void testInvalidRarerLabel() {
    CountingLearner learner = new CountingLearner();
    BalancedFitFn fitFn = new BalancedFitFn(Arrays.asList((SimpleOnlineLearner)learner));
    ShuffleFn<LabeledVector> shuffleFn = new LabelSeparatingShuffleFn(seed, -1.0);
    ParallelLearner parallelLearner = new ParallelLearner();
    PCollection<OnlineLearnerRun> pruns = parallelLearner.runPipeline(TRAINING_DATA,
        shuffleFn, new CrossfoldFn<Pair<Integer, LabeledVector>>(1, seed),
        new SimpleDistributeFn<Integer, Pair<Integer, LabeledVector>>(1, seed),
        fitFn);
  }
  
  private class CountingLearner implements SimpleOnlineLearner {
    public int numPositive;
    public int numNegative;
    
    @Override
    public OnlineLearnerParams getParams() {
      return null;
    }

    @Override
    public Classifier getClassifier() {
      return null;
    }

    @Override
    public boolean update(LabeledVector obs) {
      if (obs.getLabel() == 1.0) {
        numPositive++;
      } else {
        numNegative++;
      }
      return true;
    }
    
  }
}
