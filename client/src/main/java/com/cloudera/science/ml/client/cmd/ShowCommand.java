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
import java.util.Iterator;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.VectorInputParameters;

@Parameters(commandDescription = "Print the first few vectors from the given path to the command line")
public class ShowCommand implements Command {

  @Parameter(names = "--count",
      description = "The maximum number of vectors to print")
  private int count = 20;
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  private VectorInputParameters inputParams = new VectorInputParameters();

  @Override
  public int execute(Configuration conf) throws IOException {
    Pipeline p = pipelineParams.create(ShowCommand.class, conf);
    // Use a default header so we can read text files w/o a header
    PCollection<Vector> vectors = inputParams.getVectors(p);
    
    Iterator<Vector> iter = vectors.materialize().iterator();
    for (int i = 0; i < count && iter.hasNext(); i++) {
      System.out.println(iter.next());
    }
    return 0;
  }

  @Override
  public String getDescription() {
    return "Print the first few vectors from the given path to the command line";
  }
}
