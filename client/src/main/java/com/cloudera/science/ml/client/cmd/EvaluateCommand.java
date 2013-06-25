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

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pair;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRuns;
import com.cloudera.science.ml.classifier.core.ModelScore;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.parallel.ParallelEvaluator;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.core.vectors.LabeledVector;

@Parameters(commandDescription = "Evaluates a set of models trained by the fit command")
public class EvaluateCommand implements Command {

  @Parameter(names = "--input-file", required = true,
      description = "A local Avro file that contains models trained by the fit command")
  private String runsFile;
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    List<MLOnlineLearnerRuns> mlRuns = AvroIO.read(MLOnlineLearnerRuns.class, new File(runsFile));
    OnlineLearnerRuns runs = ClassifierAvros.toOnlineLearnerRuns(mlRuns.get(0));
    
    Pipeline p = pipelineParams.create(EvaluateCommand.class, conf);

    PCollection<LabeledVector> labeledVectors = inputParams.getLabeledVectors(p);
 
    ParallelEvaluator evaluator = new ParallelEvaluator();
    Collection<Pair<OnlineLearnerRun, ModelScore>> runsWithScores =
        evaluator.evaluateModels(runs, labeledVectors);
    
    for (Pair<OnlineLearnerRun, ModelScore> runWithScore : runsWithScores) {
      OnlineLearnerRun run = runWithScore.first();
      ModelScore score = runWithScore.second();
      System.out.println("fold: " + run.getFold()
          + "\tpartition: " + run.getPartition()
          + "\tparams: " + run.getParams()
          + "\tscore: " + score);
    }
    
    p.done();
    
    return 0;
  }

  @Override
  public String getDescription() {
    return "Evaluates a set of models trained by the fit command";
  }

}
