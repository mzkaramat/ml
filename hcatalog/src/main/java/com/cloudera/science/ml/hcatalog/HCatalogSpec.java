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

import java.util.List;

import org.apache.hcatalog.common.HCatException;
import org.apache.hcatalog.data.schema.HCatSchema;

import com.cloudera.science.ml.core.records.DataType;
import com.cloudera.science.ml.core.records.FieldSpec;
import com.cloudera.science.ml.core.records.Spec;

/**
 *
 */
public class HCatalogSpec implements Spec {

  private final HCatSchema schema;
  
  public HCatalogSpec(HCatSchema schema) {
    this.schema = schema;
  }
  
  @Override
  public DataType getDataType() {
    return DataType.RECORD;
  }

  @Override
  public int size() {
    return schema.size();
  }

  @Override
  public List<String> getFieldNames() {
    return schema.getFieldNames();
  }

  @Override
  public FieldSpec getField(int index) {
    return new HCatalogFieldSpec(schema.get(index), index);
  }

  @Override
  public FieldSpec getField(String fieldName) {
    try {
      return new HCatalogFieldSpec(schema.get(fieldName), schema.getPosition(fieldName));
    } catch (HCatException e) {
      throw new IllegalArgumentException("Invalid field name: " + fieldName, e);
    }
  }

}
