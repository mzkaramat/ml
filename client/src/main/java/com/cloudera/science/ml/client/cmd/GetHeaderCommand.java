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

import org.apache.hadoop.conf.Configuration;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.cloudera.science.ml.hcatalog.HCatalog;
import com.cloudera.science.ml.hcatalog.HCatalogSpec;
import com.google.common.base.Charsets;
import com.google.common.io.Files;

@Parameters(commandDescription = "Create a header file from a Hive table")
public class GetHeaderCommand implements Command {

  @Parameter(names = "--table", required=true,
      description = "The Hive table to create the header file for")
  private String table;
  
  @Parameter(names = "--header-file", required=true,
      description = "The name of the local file to write the header information to")
  private String headerFile;
  
  @Override
  public int execute(Configuration conf) throws IOException {
    HCatalogSpec spec = HCatalog.getSpec("default", table);
    Files.write(spec.toHeader().toString(), new File(headerFile), Charsets.UTF_8);
    return 0;
  }

  @Override
  public String getDescription() {
    return "Create a header file from a Hive table";
  }
}
