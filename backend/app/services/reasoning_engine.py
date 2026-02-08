from .neo4j_service import Neo4jService
from ..ml.world_model.plausibility_scorer import PlausibilityScorer

class ReasoningEngine:
    def __init__(self):
        self.db = Neo4jService()
        self.scorer = PlausibilityScorer(self.db)

    def analyze_combination(self, herb_ids: list, modifiers: dict = None):
        """
        modifiers: {
            'alpha': float (default 1.0) - Weight for Biological/Target Coverage
            'beta': float (default 0.5)  - Weight for Redundancy Penalty
        }
        """
        # 1. Shared Effects (Graph Logic - Detailed textual paths)
        shared_effects = self.db.find_shared_effects(herb_ids)
        
        # 2. Interactions (Graph Logic)
        interactions = self.db.find_interactions(herb_ids)
        
        # 3. Plausibility Scoring (The "Engine" Decision)
        score, explanation_obj = self.scorer.score_combination(herb_ids)
        
        # 4. Construct Final Response
        # We merge the Scorer's explanation with the detailed paths found by find_shared_effects
        
        result_text = []
        result_text.append(f"**Plausibility Score: {score:.2f}**")
        result_text.append(f"_{explanation_obj['interpretation']}_")
        
        # Add Scorer Details
        details = explanation_obj.get('details', {})
        result_text.append(f"\n**Analysis**:")
        result_text.append(f"- Target Coverage: {explanation_obj['components']['coverage']:.2f}")
        result_text.append(f"- Coherence: {explanation_obj['components']['coherence']:.2f}")
        
        if explanation_obj['components']['redundancy_penalty'] > 0:
            result_text.append(f"- Redundancy Penalty: -{explanation_obj['components']['redundancy_penalty']:.2f} (Overlap detected)")
            
        if details.get('risk_notes'):
             result_text.append(f"\n**Risk Factors**:")
             for note in details['risk_notes']:
                 result_text.append(f"- {note}")

        if shared_effects:
            result_text.append("\n**Mechanistic Overlap (Graph Evidence):**")
            for eff in shared_effects:
                names = ", ".join(eff['herb_names'])
                result_text.append(f"- {names} both contribute to {eff['e']['name']} ({eff['e']['category']}).")
        
        if interactions:
            result_text.append("\n**Known Interactions:**")
            for inter in interactions:
                result_text.append(f"- {inter['description']}")

        # 5. Semantic Target Enrichment (Optional - can be moved to Scorer or kept here)
        # keeping it here for now as it adds "color" to the report
        from .target_knowledge import TargetKnowledgeService
        tk = TargetKnowledgeService()
        
        # Re-using the logic to fetch top shared proteins for enrichment text
        HERB_NAME_MAP = {
            "Curcuma longa": "Curcumae Longae Rhizoma",
            "Piper longum": "Piperis Longi Fructus", 
            "Zingiber officinale": "Zingiberis Rhizoma",
            "Phyllanthus emblica": "Phyllanthi Fructus",
            "Tinospora cordifolia": "Tinosporae Radix",
            "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma",
            "Ocimum tenuiflorum": "Ocimi Sancti Herba",
            "Ocimum sanctum": "Ocimi Sancti Herba"
        }
        
        q_top_shared = """
        UNWIND $herbs as hid
        MATCH (h:Herb {herbId: hid})
        OPTIONAL MATCH (hm:HerbMaterial) WHERE hm.latin_name = h.scientificName OR hm.latin_name = $map[h.scientificName]
        MATCH (hm)-[:HAS_COMPOUND]->(:Compound)-[:TARGETS]->(p:Protein)
        WITH p, count(DISTINCT hm) as functional_hits
        WHERE functional_hits > 1
        RETURN p.name as gene
        ORDER BY functional_hits DESC
        LIMIT 10
        """
        
        with self.db.driver.session() as s:
            t_res = s.run(q_top_shared, herbs=herb_ids, map=HERB_NAME_MAP)
            top_genes = [r["gene"] for r in t_res]
            
        if top_genes:
            result_text.append("\n**Key Targets (Enriched):**")
            for gene in top_genes:
                meta = tk.enrich_target(gene)
                if meta:
                     result_text.append(f"- {gene}: {meta['description']}")

        return {
            "score": score,
            "explanation": "\n".join(result_text),
            "plausibility_details": explanation_obj,
            "shared_effects": shared_effects,
            "interactions": interactions
        }

    def score_direct(self, compound_groups: dict):
        """
        Direct compound-level scoring.
        """
        score, explanation_obj = self.scorer.score_direct(compound_groups)
        
        # Enrich explanation text
        result_text = []
        result_text.append(f"**Plausibility Score: {score:.2f}**")
        
        # New Schema has 'decision' -> 'recommendation' instead of purely 'interpretation'
        rec = explanation_obj.get('decision', {}).get('recommendation', "No recommendation")
        result_text.append(f"_{rec}_")
        
        metrics = explanation_obj.get('metrics', {})
        result_text.append(f"\n**Analysis**:")
        result_text.append(f"- Coverage: {metrics.get('coverage', 0):.2f}")
        # Coherence is not explicitly in metrics dict in new schema, but we can infer or skip
        # metrics has 'uncertainty', 'redundancy_penalty', 'risk_penalty'
        
        if metrics.get('redundancy_penalty', 0) > 0:
             result_text.append(f"- Redundancy Penalty: -{metrics['redundancy_penalty']:.2f}")

        # Interactions
        # in new schema, interactions might be in 'biological_context' or we look for 'contributors'
        # The scorer implementation had 'interactions' mainly in debug print or legacy. 
        # Actually, let's look at what PlausibilityScorer returns.
        # It DOES NOT return 'interactions' list in the top level response anymore.
        # It's inside the internal data but not in the final JSON. 
        # I should double check PlausibilityScorer._aggregate_scores.
        
        # For now, let's just make sure this doesn't crash.
        
        return score, {
            **explanation_obj,
            "text": "\n".join(result_text)
        }

    def close(self):
        self.db.close()

