// Constraints
CREATE CONSTRAINT herb_latin IF NOT EXISTS FOR (h:HerbMaterial) REQUIRE h.latin_name IS UNIQUE;
CREATE CONSTRAINT compound_id IF NOT EXISTS FOR (c:Compound) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT compound_inchikey IF NOT EXISTS FOR (c:Compound) REQUIRE c.inchikey IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT formula_name IF NOT EXISTS FOR (f:Formula) REQUIRE f.name IS UNIQUE;

// 1. Herbs
LOAD CSV WITH HEADERS FROM 'file:///nodes_herb_material.csv' AS row
MERGE (h:HerbMaterial {latin_name: row.latin_name})
SET h.common = row.common_name,
    h.chinese = row.chinese_name,
    h.korean = row.korean_name,
    h.pinyin = row.pinyin;

// 2. Compounds (Authoritative with Chemical Properties)
LOAD CSV WITH HEADERS FROM 'file:///nodes_compound_properties.csv' AS row
MERGE (c:Compound {compoundId: row.compound_id})
SET c.name = CASE WHEN row.name IS NOT NULL THEN row.name ELSE row.compound_id END,
    c.inchikey = row.inchikey,
    c.inchi = row.inchi,
    c.smiles = row.smiles,
    c.cid = row.cid,
    c.formula = row.formula,
    c.mw = toFloat(row.mw),
    c.exact_mw = toFloat(row.exact_mw),
    c.logp = toFloat(row.logp),
    c.tpsa = toFloat(row.tpsa),
    c.hba = toInteger(row.hba),
    c.hbd = toInteger(row.hbd),
    c.rotb = toInteger(row.rotb),
    c.atom_count = toInteger(row.atom_count),
    c.nring = toInteger(row.nring),
    c.property_source = row.property_source,
    c.confidence = row.confidence;

// 3. Herb -> Compound
LOAD CSV WITH HEADERS FROM 'file:///edges_herb_compound.csv' AS row
MATCH (h:HerbMaterial {latin_name: row.herb_latin})
MATCH (c:Compound {compoundId: row.compound_id})
MERGE (h)-[:HAS_COMPOUND]->(c);

// 4. Proteins
LOAD CSV WITH HEADERS FROM 'file:///nodes_protein.csv' AS row
MERGE (p:Protein {id: row.protein_id})
SET p.name = row.protein_name;

// 5. Compound -> Protein
LOAD CSV WITH HEADERS FROM 'file:///edges_compound_protein.csv' AS row
MATCH (c:Compound {compoundId: row.compound_id})
MATCH (p:Protein {id: row.protein_id})
MERGE (c)-[r:TARGETS]->(p)
SET r.score = row.score;

// 6. Diseases
LOAD CSV WITH HEADERS FROM 'file:///nodes_disease.csv' AS row
MERGE (d:Disease {id: row.disease_id})
SET d.name = row.disease_name;

// 7. Protein -> Disease
LOAD CSV WITH HEADERS FROM 'file:///edges_protein_disease.csv' AS row
MATCH (p:Protein {id: row.protein_id})
MATCH (d:Disease {id: row.disease_id})
MERGE (p)-[r:ASSOCIATED_WITH]->(d)
SET r.score = row.score;

// 8. Formulas
LOAD CSV WITH HEADERS FROM 'file:///nodes_formula.csv' AS row
MERGE (f:Formula {name: row.formula_name})
SET f.source_book = row.source_book;

// 9. Formula -> Herb/Ingredient
LOAD CSV WITH HEADERS FROM 'file:///edges_formula_herb.csv' AS row
MATCH (f:Formula {name: row.formula_name})
MATCH (h:HerbMaterial {latin_name: row.ingredient_latin})
MERGE (f)-[r:HAS_INGREDIENT]->(h)
SET r.dosage = row.dosage,
    r.unit = row.unit,
    r.seq = row.seq,
    r.process = row.process;
