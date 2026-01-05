from .neo4j_service import Neo4jService

class ReasoningEngine:
    def __init__(self):
        self.db = Neo4jService()

    def analyze_combination(self, herb_ids: list):
        # 1. Shared Effects
        shared_effects = self.db.find_shared_effects(herb_ids)
        
        # 2. Interactions
        interactions = self.db.find_interactions(herb_ids)
        
        # 3. Explanation Construction
        explanation = []
        if shared_effects:
            explanation.append("Synergy found: The selected herbs share common biological effects.")
            for eff in shared_effects:
                names = ", ".join(eff['herb_names'])
                explanation.append(f"- {names} both contribute to {eff['e']['name']} ({eff['e']['category']}).")
        
        if interactions:
            explanation.append("\nNote on Interactions:")
            for inter in interactions:
                explanation.append(f"- {inter['description']}")
        
        return {
            "shared_effects": shared_effects,
            "interactions": interactions,
            "explanation": "\n".join(explanation)
        }

    def close(self):
        self.db.close()
