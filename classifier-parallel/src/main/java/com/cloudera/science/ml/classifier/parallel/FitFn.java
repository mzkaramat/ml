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

import org.apache.crunch.DoFn;
import org.apache.crunch.Pair;

import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.core.vectors.LabeledVector;

/**
 * A reduce-side function for fitting a set of learners to data.  The key is a
 * crossfold-partition pair and the values are vectors with an integer to
 * order them.
 */
public abstract class FitFn extends DoFn<Pair<Pair<Integer, Integer>, Iterable<Pair<Integer, LabeledVector>>>, OnlineLearnerRun> {

}
