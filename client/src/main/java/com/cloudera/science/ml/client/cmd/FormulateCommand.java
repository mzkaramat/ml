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
import java.util.List;

import org.apache.crunch.PCollection;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.math.Vector;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.RecordInputParameters;
import com.cloudera.science.ml.client.params.SummaryParameters;
import com.cloudera.science.ml.client.params.VectorOutputParameters;
import com.cloudera.science.ml.core.formula.Formula;
import com.cloudera.science.ml.core.formula.Term;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.parallel.formula.Formulator;
import com.cloudera.science.ml.parallel.records.SummarizedRecords;
import com.cloudera.science.ml.parallel.types.MLAvros;
import com.google.common.collect.Lists;

@Parameters(commandDescription = "Converts records into a distributed matrix using an R-style formula string")
public class FormulateCommand implements Command {
  
  @Parameter(names = "--summary-file", required=true,
      description = "The name of a local JSON file that contains the summary info to use for normalizing the data")
  private String summaryFile;
  
  @Parameter(names = "--output-path", required = true,
      description = "The name of the output path, which will overwrite any existing files with that name")
  private String outputPath;
  
  @Parameter(names = "--formula", variableArity=true, required=true,
      description = "An R-style formula that describes how to transform the records into vectors")
  private List<String> formulaTerms = Lists.newArrayList();
  
  @ParametersDelegate
  private RecordInputParameters inputParams = new RecordInputParameters();

  @ParametersDelegate
  private VectorOutputParameters outputParams = new VectorOutputParameters();
  
  @ParametersDelegate
  private SummaryParameters summaryParams = new SummaryParameters();
  
  @ParametersDelegate
  private PipelineParameters pipelineParams = new PipelineParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    Summary summary = summaryParams.get(summaryFile);
    List<Term> terms = parseTerms(summary);
    Formula formula = Formula.compile(terms, summary);
    Formulator formulator = new Formulator(formula);
    
    Pipeline p = pipelineParams.create(Formulator.class, conf);
    SummarizedRecords records = inputParams.getSummarizedRecords(p, summary);
    PCollection<Vector> out = formulator.apply(records.get());
    outputParams.writeVectors(out, outputPath, MLAvros.vector());
    p.done();
    return 0;
  }

  private List<Term> parseTerms(Summary summary) {
    List<Term> terms = Lists.newArrayList();
    //TODO, obviously
    return terms;
  }

  @Override
  public String getDescription() {
    return "Converts records into a distributed matrix using an R-style formula string";
  }

}
