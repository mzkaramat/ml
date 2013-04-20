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

import org.apache.hcatalog.common.HCatException;
import org.apache.hcatalog.data.DefaultHCatRecord;
import org.apache.hcatalog.data.HCatRecord;
import org.apache.hcatalog.data.schema.HCatSchema;

import com.cloudera.science.ml.core.records.Record;
import com.cloudera.science.ml.core.records.Spec;

public class HCatalogRecord implements Record {

  private final HCatRecord impl;
  private final HCatSchema schema;
  
  public HCatalogRecord(HCatRecord impl, HCatSchema schema) {
    this.impl = impl;
    this.schema = schema;
  }
  
  public HCatRecord getImpl() {
    return impl;
  }
  
  public HCatSchema getSchema() {
    return schema;
  }
  
  @Override
  public Spec getSpec() {
    return new HCatalogSpec(schema);
  }

  @Override
  public int size() {
    return impl.size();
  }
  
  @Override
  public Record copy(boolean deep) {
    if (deep) {
      throw new UnsupportedOperationException("HCatalog does not support deep copies yet");
    }
    HCatRecord copy = new DefaultHCatRecord();
    try {
      copy.copy(impl);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
    return new HCatalogRecord(copy, schema);
  }

  @Override
  public Object get(int index) {
    return impl.get(index);
  }

  @Override
  public String getAsString(int index) {
    return get(index).toString();
  }

  @Override
  public double getAsDouble(int index) {
    Object ret = get(index);
    if (ret instanceof Number) {
      return ((Number) ret).doubleValue();
    } else {
      try {
        return Double.valueOf(ret.toString());
      } catch (NumberFormatException ignored) {
        return Double.NaN;
      }
    }
  }

  @Override
  public Boolean getBoolean(int index) {
    return getBoolean(schema.getFieldNames().get(index));
  }

  @Override
  public Boolean getBoolean(String fieldName) {
    try {
      return impl.getBoolean(fieldName, schema);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Double getDouble(int index) {
    return getDouble(schema.getFieldNames().get(index));
  }

  @Override
  public Double getDouble(String fieldName) {
    try {
      return impl.getDouble(fieldName, schema);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Integer getInteger(int index) {
    return getInteger(schema.getFieldNames().get(index));
  }

  @Override
  public Integer getInteger(String fieldName) {
    try {
      return impl.getInteger(fieldName, schema);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Long getLong(int index) {
    return getLong(schema.getFieldNames().get(index));
  }

  @Override
  public Long getLong(String fieldName) {
    try {
      return impl.getLong(fieldName, schema);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public String getString(int index) {
    return getString(schema.getFieldNames().get(index));
  }

  @Override
  public String getString(String fieldName) {
    try {
      return impl.getString(fieldName, schema);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Record set(int index, Object value) {
    impl.set(index, value);
    return this;
  }

  @Override
  public Record set(String fieldName, Object value) {
    try {
      impl.set(fieldName, schema, value);
    } catch (HCatException e) {
      throw new RuntimeException(e);
    }
    return this;
  }

  @Override
  public String toString() {
    return impl.toString();
  }
}
