// Import Chemical Properties from nodes_compound_properties.csv

// Constraints
CREATE CONSTRAINT compound_inchikey IF NOT EXISTS FOR (c:Compound) REQUIRE c.inchikey IS UNIQUE;
CREATE CONSTRAINT compound_cid IF NOT EXISTS FOR (c:Compound) REQUIRE c.cid IS UNIQUE;

// Load CSV
LOAD CSV WITH HEADERS FROM 'file:///nodes_compound_properties.csv' AS row
MERGE (c:Compound {compound_id: row.compound_id})
SET
  c.inchikey      = row.inchikey,
  c.inchi         = row.inchi,
  c.smiles        = row.smiles,
  c.cid           = row.cid,
  c.formula       = row.formula,
  c.mw            = toFloat(row.mw),
  c.exact_mw      = toFloat(row.exact_mw),
  c.logp          = toFloat(row.logp),
  c.tpsa          = toFloat(row.tpsa),
  c.hba           = toInteger(row.hba),
  c.hbd           = toInteger(row.hbd),
  c.rotb          = toInteger(row.rotb),
  c.atom_count    = toInteger(row.atom_count),
  c.nring         = toInteger(row.nring),
  c.property_source = row.property_source,
  c.confidence    = row.confidence,
  c.source_id     = row.source_id;
