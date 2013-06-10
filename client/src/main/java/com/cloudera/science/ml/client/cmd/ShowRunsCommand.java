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
import java.util.List;

import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRun;
import com.cloudera.science.ml.classifier.core.OnlineLearnerRuns;
import com.cloudera.science.ml.classifier.parallel.types.ClassifierAvros;
import com.cloudera.science.ml.client.util.AvroIO;
import com.cloudera.science.ml.classifier.avro.MLOnlineLearnerRuns;

@Parameters(commandDescription = "Print the first few vectors from the given path to the command line")
public class ShowRunsCommand implements Command {

  @Parameter(names = "--input-file", required = true,
      description = "A local Avro file that contains  models trained by the fit command")
  private String runsFile;
  
  @Override
  public int execute(Configuration conf) throws IOException {
    List<MLOnlineLearnerRuns> mlRuns = AvroIO.read(MLOnlineLearnerRuns.class, new File(runsFile));
    OnlineLearnerRuns runs = ClassifierAvros.toOnlineLearnerRuns(mlRuns.get(0));
    
    System.out.println("seed: " + runs.getSeed());
    System.out.println("num folds: " + runs.getNumFolds());
    
    for (OnlineLearnerRun run : runs.getRuns()) {
      System.out.println(run);
      System.out.println("weight vector: " + run.getClassifier().getWeights());
    }
    return 0;
  }

  @Override
  public String getDescription() {
    return "Print the first few vectors from the given path to the command line";
  }
}
