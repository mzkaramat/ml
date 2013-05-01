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

import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.cloudera.science.ml.client.params.PipelineParameters;
import com.cloudera.science.ml.client.params.RecordInputParameters;
import com.cloudera.science.ml.client.params.RecordOutputParameters;
import com.cloudera.science.ml.client.params.SummaryParameters;
import com.cloudera.science.ml.core.records.Spec;
import com.cloudera.science.ml.core.records.Specs;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.parallel.pivot.Pivot;
import com.cloudera.science.ml.parallel.records.Records;
import com.cloudera.science.ml.parallel.records.SummarizedRecords;
import com.google.common.collect.Lists;

@Parameters(commandDescription = "Performs a pivot operation to convert a 'long' table into a 'wide' table")
public class PivotCommand implements Command {

  @Parameter(names = "--summary-file", required=true,
      description = "The local summary stats created by the summary command")
  String summaryFile;
  
  @Parameter(names = "--id-fields", required=true,
      splitter = CommaParameterSplitter.class,
      description = "The fields in the long format that identify a group of records that belong in the same row of the wide format")
  private List<String> idFields = Lists.newArrayList();
  
  @Parameter(names = "--value-fields", required=true,
      splitter = CommaParameterSplitter.class,
      description = "The time-varying values in the long format that correspond to individual columns in the wide format")
  private List<String> valueFields = Lists.newArrayList();
  
  @Parameter(names = "--var-field", required=true,
      description = "The variable that differentiates multiple records from the same group/individual in the long format")
  private String varField;
  
  @Parameter(names = "--aggregation",
      description = "The type of strategy to use for aggregating values (either SUM or MEAN)")
  private String aggregation = "SUM";
  
  @Parameter(names = "--output-path", required=true,
      description = "Where to write the pivoted output records in HDFS")
  private String outputPath;
  
  @ParametersDelegate
  RecordInputParameters inputParams = new RecordInputParameters();
  
  @ParametersDelegate
  RecordOutputParameters outputParams = new RecordOutputParameters();
  
  @ParametersDelegate
  PipelineParameters pipelineParams = new PipelineParameters();
  
  @ParametersDelegate
  SummaryParameters summaryParams = new SummaryParameters();
  
  @Override
  public int execute(Configuration conf) throws IOException {
    Summary summary = summaryParams.get(summaryFile);
    Pipeline p = pipelineParams.create(PivotCommand.class, conf);
    SummarizedRecords records = inputParams.getSummarizedRecords(p, summary);
    
    Spec spec = summary.getSpec();
    List<Integer> idColumns = Specs.getFieldIds(spec, idFields);
    Integer timeColumn = Specs.getFieldId(spec, varField);
    List<Integer> valueColumns = Specs.getFieldIds(spec, valueFields);
    Pivot.Agg agg = Pivot.Agg.valueOf(aggregation);
    
    Records pivoted = Pivot.pivot(records, idColumns, timeColumn, valueColumns, agg);
    outputParams.writeRecords(pivoted.get(), pivoted.getSpec(), outputPath);
    p.done();
    return 0;
  }

  @Override
  public String getDescription() {
    return "Performs a pivot operation that converts a 'long' table into a 'wide' table";
  }

}
