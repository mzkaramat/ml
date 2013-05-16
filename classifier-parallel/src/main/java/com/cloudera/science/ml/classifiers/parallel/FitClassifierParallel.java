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

package com.cloudera.science.ml.classifiers.parallel;

import java.util.ArrayList;
import java.util.List;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.PCollection;
import org.apache.mahout.math.Vector;

import com.cloudera.science.ml.classifiers.ClassifierParams;
import com.cloudera.science.ml.classifiers.ClassifierTrainer;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.types.MLAvros;

public class FitClassifierParallel {
  public void fitClassifier(ClassifierParams params, long seed,
      PCollection<LabeledVector> trainingData) {
    FitClassifierFn<LabeledVector> fn = new FitClassifierFn<LabeledVector>(params, seed);
    trainingData.parallelDo("fit-classifier", fn, MLAvros.vector());
  }
  
  private static class FitClassifierFn<D extends LabeledVector> extends DoFn<LabeledVector, Vector> {
    private static final long serialVersionUID = 5123451370159638934L;
    
    private ClassifierParams params;
    private long seed;
    private final List<LabeledVector> trainingSet = new ArrayList<LabeledVector>();
    
    public FitClassifierFn(ClassifierParams params, long seed) {
      this.params = params;
      this.seed = seed;
    }
    
    @Override
    public void process(LabeledVector trainingVec, Emitter<Vector> emitter) {
      trainingSet.add(trainingVec);
    }
    
    @Override
    public void cleanup(Emitter<Vector> emitter) {
      ClassifierTrainer classifiers = new ClassifierTrainer(seed);
      classifiers.fitClassifier(trainingSet, params);
    }
  }
}
