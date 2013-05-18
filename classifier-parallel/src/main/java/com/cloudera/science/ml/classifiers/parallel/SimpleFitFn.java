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

import java.util.List;

import org.apache.crunch.DoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

import com.cloudera.science.ml.classifier.core.Classifier;
import com.cloudera.science.ml.classifier.core.OnlineLearner;
import com.cloudera.science.ml.classifier.simple.SimpleOnlineLearner;
import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.base.Pairs;
import com.google.common.base.Function;
import com.google.common.collect.Lists;

/**
 *
 */
public class SimpleFitFn<K, SK> extends DoFn<Pair<K, Iterable<Pair<SK, LabeledVector>>>, Pair<K, List<Classifier>>> {
  @Override
  public void process(Pair<K, Iterable<Pair<SK, LabeledVector>>> in,
      Emitter<Pair<K, List<Classifier>>> emitter) {
    List<SimpleOnlineLearner> learners = Lists.newArrayList(); // TODO
    for (LabeledVector obs : Pairs.second(in.second())) {
      for (SimpleOnlineLearner learner : learners) {
        learner.update(obs);
      }
    }
    emitter.emit(Pair.of(in.first(), Lists.transform(learners, new Function<OnlineLearner, Classifier>() {
      @Override
      public Classifier apply(OnlineLearner learner) {
        return learner.getClassifier();
      }
    })));
  } 

}
