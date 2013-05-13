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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class HCatalogTest {
  @Test
  public void testTableParse() throws Exception {
    String tbl = "foo";
    assertEquals("default", HCatalog.getDbName(tbl));
    assertEquals("foo", HCatalog.getTableName(tbl));
  }
  
  @Test
  public void testDbTableParse() throws Exception {
    String tbl = "foo.bar";
    assertEquals("foo", HCatalog.getDbName(tbl));
    assertEquals("bar", HCatalog.getTableName(tbl));
  }
  
  @Test
  public void testTwoDotDbTableParse() throws Exception {
    String tbl = "foo.bar.baz";
    assertEquals("foo", HCatalog.getDbName(tbl));
    assertEquals("bar.baz", HCatalog.getTableName(tbl));
  }
}
