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
package com.cloudera.science.ml.hcatalog;

import org.apache.hcatalog.data.schema.HCatFieldSchema;
import org.apache.hcatalog.data.schema.HCatSchema;

import com.cloudera.science.ml.core.records.FieldSpec;
import com.cloudera.science.ml.core.records.RecordSpec;
import com.google.common.base.Function;
import com.google.common.collect.Lists;

public class HCatalogSpec extends RecordSpec {
  private final HCatSchema schema;
  
  public HCatalogSpec(final HCatSchema schema) {
    super(Lists.transform(schema.getFields(), new Function<HCatFieldSchema, FieldSpec>() {
      @Override
      public FieldSpec apply(HCatFieldSchema fs) {
        return new HCatalogFieldSpec(fs, schema.getPosition(fs.getName()));
      }
    }));
    this.schema = schema;
  }
  
  public HCatSchema getImpl() {
    return schema;
  }
}
