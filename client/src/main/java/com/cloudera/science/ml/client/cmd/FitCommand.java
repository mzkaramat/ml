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

package com.cloudera.science.ml.client.cmd;

import java.io.IOException;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.classifiers.ClassifierParams;
import com.cloudera.science.ml.classifiers.EtaType;
import com.cloudera.science.ml.classifiers.LearnerType;
import com.cloudera.science.ml.classifiers.LoopType;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.core.vectors.LabeledVector;

public class FitCommand implements Command {
  
  @Parameter(names = "--training-vectors-path",
      description = "A path that contains the vector(s) used for initializing k-means||")
  private String initVectorsPath;
  
  @Parameter(names = "--loop-type",
      description = "The loop strategy for training the classifier")
  private String loopType;
  
  @Parameter(names = "--learner-type",
      description = "The kind of classifier to train")
  private String learnerType;
  
  @Parameter(names = "--eta-type",
      description = "The type of eta to use in the SGD")
  private String etaType;

  @Parameter(names = "--num-iters",
      description = "The number of SGD iterations to run in each mapper")
  private int numIters;
  
  @Parameter(names = "--num-features",
      description = "The number of features")
  private int numFeatures;
  
  @Parameter(names = "--lambda",
      description = "The regularization parameter")
  private float lambda;

  @Parameter(names = "--c",
      description = "Maximum size of any step taken in a single passive-aggressive update")
  private float c;

  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(FitCommand.class, conf);
    ClassifierParams params = new ClassifierParams(
        LearnerType.valueOf(learnerType),
        LoopType.valueOf(loopType),
        EtaType.valueOf(etaType),
        lambda, c, numIters, numFeatures);

    PCollection<LabeledVector> labeledVectors = inputParams.getLabeledVectors(p);
    
    return 0;
  }
  
  @Override
  public String getDescription() {
    return "Fit a classification model to a training set";
  }
}
