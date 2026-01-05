// Constraints already created
// CREATE CONSTRAINT FOR (h:Herb) REQUIRE h.herbId IS UNIQUE IF NOT EXISTS; ...

// Load Herbs
LOAD CSV WITH HEADERS FROM 'file:///nodes_herb.csv' AS row
MERGE (h:Herb {herbId: row['herbId:ID']})
SET h.name = row.name,
    h.scientificName = row.scientificName,
    h.description = row.description,
    h.sanskritName = row.sanskritName;

// Load Compounds
LOAD CSV WITH HEADERS FROM 'file:///nodes_compound.csv' AS row
MERGE (c:Compound {compoundId: row['compoundId:ID']})
SET c.name = row.name,
    c.chemicalClass = row.chemicalClass;

// Load Effects
LOAD CSV WITH HEADERS FROM 'file:///nodes_effect.csv' AS row
MERGE (e:BiologicalEffect {effectId: row['effectId:ID']})
SET e.name = row.name,
    e.category = row.category;

// Load Evidence
LOAD CSV WITH HEADERS FROM 'file:///nodes_evidence.csv' AS row
MERGE (ev:Evidence {evidenceId: row['evidenceId:ID']})
SET ev.title = row.title,
    ev.url = row.url,
    ev.doi = row.doi;

// Load CONTAINS
LOAD CSV WITH HEADERS FROM 'file:///edges_contains.csv' AS row
MATCH (h:Herb {herbId: row[':START_ID']})
MATCH (c:Compound {compoundId: row[':END_ID']})
MERGE (h)-[r:CONTAINS]->(c)
SET r.concentration = row.concentration;

// Load MODULATES
LOAD CSV WITH HEADERS FROM 'file:///edges_modulates.csv' AS row
MATCH (c:Compound {compoundId: row[':START_ID']})
MATCH (e:BiologicalEffect {effectId: row[':END_ID']})
MERGE (c)-[r:MODULATES]->(e)
SET r.direction = row.direction,
    r.strength = row.strength;

// Load SUPPORTED_BY (Herb)
LOAD CSV WITH HEADERS FROM 'file:///edges_supported_by.csv' AS row
MATCH (s:Herb {herbId: row[':START_ID']})
MATCH (ev:Evidence {evidenceId: row[':END_ID']})
MERGE (s)-[r:SUPPORTED_BY]->(ev)
SET r.confidence = row.confidence;

// Load SUPPORTED_BY (Compound)
LOAD CSV WITH HEADERS FROM 'file:///edges_supported_by.csv' AS row
MATCH (s:Compound {compoundId: row[':START_ID']})
MATCH (ev:Evidence {evidenceId: row[':END_ID']})
MERGE (s)-[r:SUPPORTED_BY]->(ev)
SET r.confidence = row.confidence;

// Load SUPPORTED_BY (Effect)
LOAD CSV WITH HEADERS FROM 'file:///edges_supported_by.csv' AS row
MATCH (s:BiologicalEffect {effectId: row[':START_ID']})
MATCH (ev:Evidence {evidenceId: row[':END_ID']})
MERGE (s)-[r:SUPPORTED_BY]->(ev)
SET r.confidence = row.confidence;

// Load ENHANCES_ABSORPTION
LOAD CSV WITH HEADERS FROM 'file:///edges_enhances_absorption.csv' AS row
MATCH (c1:Compound {compoundId: row[':START_ID']})
MATCH (c2:Compound {compoundId: row[':END_ID']})
MERGE (c1)-[r:ENHANCES_ABSORPTION]->(c2)
SET r.mechanism = row.mechanism;
