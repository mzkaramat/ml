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

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.beust.jcommander.ParametersDelegate;
import com.beust.jcommander.converters.BooleanConverter;
import com.beust.jcommander.converters.CommaParameterSplitter;
import com.beust.jcommander.converters.DoubleConverter;
import com.beust.jcommander.converters.IntegerConverter;
import com.cloudera.science.ml.client.params.*;
import com.cloudera.science.ml.core.records.Header;
import com.cloudera.science.ml.core.summary.Numeric;
import com.cloudera.science.ml.core.summary.Summary;
import com.cloudera.science.ml.core.summary.SummaryBuilder;
import com.cloudera.science.ml.parallel.records.Records;
import com.cloudera.science.ml.parallel.records.SummarizedRecords;
import com.google.common.collect.Lists;
import org.apache.crunch.Pipeline;
import org.apache.hadoop.conf.Configuration;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Parameters(commandDescription = "Discretizes a numeric field of a collection of vectors by given bin size, if desired with equal frequency")
public class DiscretizeCommand implements Command {

  @Parameter(names = "--summary-file",
     description = "The local summary stats created by the summary command")
  private String summaryFile;

  @Parameter(names = "--header-file",
     description = "The header file (for CSV inputs), used if the --summary-file doesn't exist")
  private String headerFile;

  @Parameter(names = "--min-values",
     splitter = CommaParameterSplitter.class,
     converter = DoubleConverter.class,
     description = "Minimum values for numeric fields to be discretized")
  private List<Integer> minValues;

  @Parameter(names = "--max-values",
     splitter = CommaParameterSplitter.class,
     converter = DoubleConverter.class,
     description = "Maximum values for numeric fields to be discretized")
  private String maxValues;

  @Parameter(names = "--num-fields", required = true,
     description = "Column ids of values to discretize",
     splitter = CommaParameterSplitter.class,
     converter = IntegerConverter.class)
  private List<String> numFields = Lists.newArrayList();

  @Parameter(names = "--bins", required = true,
     description = "Number of bins to be created (Number of unique categories)"
  )
  private int bins;

  @Parameter(names = "--equal-freq", required = false,
     description = "If set true, tries to keep number of instances per category equal",
     converter = BooleanConverter.class)
  private boolean equalFrequency = false;

  @Parameter(names = "--output-path", required = true,
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
    Pipeline p = pipelineParams.create(DiscretizeCommand.class, conf);
    SummarizedRecords records = null;
    if (summaryFile != null) {
      Summary summary = summaryParams.get(summaryFile);
      records = inputParams.getSummarizedRecords(p, summary);
    } else {

    }

    return 0;  //To change body of implemented methods use File | Settings | File Templates.
  }

  @Override
  public String getDescription() {
    return "Discretizes a numeric field of a collection of vectors by given bin size, if desired with equal frequency";
  }
}
