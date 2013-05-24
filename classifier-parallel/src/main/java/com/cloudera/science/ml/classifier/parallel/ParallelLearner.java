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


import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;

import org.apache.crunch.PCollection;
import org.apache.crunch.PTable;
import org.apache.crunch.Pair;
import org.apache.crunch.lib.SecondarySort;
import org.apache.crunch.types.PType;
import org.apache.crunch.types.PTypeFamily;

import com.cloudera.science.ml.core.vectors.LabeledVector;
import com.cloudera.science.ml.parallel.crossfold.CrossfoldFn;
import com.cloudera.science.ml.parallel.distribute.DistributeFn;
import com.cloudera.science.ml.parallel.fn.ShuffleFn;

/**
 * Contains the core pipeline for parallel learning, which fits into a single
 * MapReduce job.  On the map side, a labeled dataset is portioned into
 * crossfolds for cross-validation, shuffled, and partitioned into smaller sets
 * for parallel processing.  On the reduce-side, the input vectors in each
 * fold-partition are used to train a set of classifiers.
 */
public class ParallelLearner {
  public PCollection<OnlineLearnerRun> runPipeline(PCollection<LabeledVector> vectors,
      ShuffleFn<LabeledVector> shuffleFn, CrossfoldFn<Pair<Integer, LabeledVector>> crossfoldFn,
      DistributeFn<Integer, Pair<Integer, LabeledVector>> distributeFn, FitFn fitFn) {
    
    PType<LabeledVector> vectype = vectors.getPType();
    PTypeFamily ptf = vectype.getFamily();
    
    // Shuffle the input vectors to determine the random order that the model
    // fitters will see them in.
    PCollection<Pair<Integer, LabeledVector>> shuffled =
        vectors.parallelDo(shuffleFn, ptf.pairs(ptf.ints(), vectype));
    
    // For cross-validation, assign each vector to its crossfolds.
    PCollection<Pair<Integer, Pair<Integer, LabeledVector>>> crossfolded =
        shuffled.parallelDo(crossfoldFn, ptf.pairs(ptf.ints(), ptf.pairs(ptf.ints(), vectype)));
    
    // Within each crossfold, apportion vectors between model-fitting runs to make
    // the size of each run more manageable.
    PTable<Pair<Integer, Integer>, Pair<Integer, LabeledVector>> apportioned =
        crossfolded.parallelDo(distributeFn,
            ptf.tableOf(ptf.pairs(ptf.ints(), ptf.ints()), ptf.pairs(ptf.ints(), vectype)));
    
    // Shuffle, sort, and train all the classifiers
    PCollection<OnlineLearnerRun> classifierRuns =
        SecondarySort.sortAndApply(apportioned, fitFn, ClassifierAvros.onlineLearnerRun());
    
    return classifierRuns;    
  }
}
