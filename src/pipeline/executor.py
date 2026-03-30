import json
from collections import defaultdict

class KBExecutor:
    def __init__(self, triples_file, entities_file):
        self.entities = {}
        self.triples = []
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

        with open(entities_file) as f:
            self.entities = json.load(f)

        with open(triples_file) as f:
            for line in f:
                t = json.loads(line)
                self.triples.append(t)
                self.outgoing[t["subject"]].append(t)
                self.incoming[t["object"]].append(t)

        # Build name index for entity resolution
        self.name_index = {}
        for eid, einfo in self.entities.items():
            name_lower = einfo["name"].lower().strip()
            self.name_index[name_lower] = eid

        print(f"KB loaded: {len(self.entities)} entities, {len(self.triples)} triples")

    def resolve_entity(self, name):
        name_lower = name.lower().strip()
        if name_lower in self.name_index:
            return self.name_index[name_lower]
        for ename, eid in self.name_index.items():
            if name_lower in ename or ename in name_lower:
                return eid
        return None

    def execute_1hop(self, entity_id, relation):
        results = []
        for triple in self.outgoing.get(entity_id, []):
            if triple["predicate"] == relation:
                target_id = triple["object"]
                target_info = self.entities.get(target_id, {})
                results.append({
                    "id": target_id,
                    "name": target_info.get("name", target_id),
                    "type": target_info.get("type", "Unknown"),
                    "triple": triple
                })
        return results

    def execute_2hop(self, entity_id, relation1, relation2):
        intermediate = self.execute_1hop(entity_id, relation1)
        results = []
        for mid in intermediate:
            targets = self.execute_1hop(mid["id"], relation2)
            for t in targets:
                t["via"] = mid["name"]
                results.append(t)
        return results

    def execute_count(self, entity_id, relation, filters=None):
        results = self.execute_1hop(entity_id, relation)
        if filters:
            for key, value in filters.items():
                results = [
                    r for r in results
                    if str(self.entities.get(r["id"], {}).get(key, "")).lower() == str(value).lower()
                ]
        return len(results)

    def execute_parsed_query(self, parsed_query):
        entity_id = self.resolve_entity(parsed_query["entity"])
        if not entity_id:
            return {"error": f"Entity not found: {parsed_query['entity']}", "results": []}

        relations = parsed_query.get("relations", [])
        query_type = parsed_query.get("type", "1hop")

        if query_type == "1hop" and len(relations) >= 1:
            results = self.execute_1hop(entity_id, relations[0])
        elif query_type == "2hop" and len(relations) >= 2:
            results = self.execute_2hop(entity_id, relations[0], relations[1])
        elif query_type == "count":
            count = self.execute_count(entity_id, relations[0], parsed_query.get("filters"))
            results = [{"name": str(count), "type": "Count"}]
        else:
            results = []

        return {
            "entity_resolved": self.entities.get(entity_id, {}).get("name", entity_id),
            "results": results,
            "count": len(results)
        }

# Test it
if __name__ == "__main__":
    executor = KBExecutor("data/kb/triples.jsonl", "data/kb/entities.json")

    # Test 1hop
    print("\nTest 1: Who authored papers about transformers?")
    eid = executor.resolve_entity("Attention is All you Need")
    if eid:
        results = executor.execute_1hop(eid, "authored_by")
        for r in results[:5]:
            print(f"  -> {r['name']}")

    # Test count
    print("\nTest 2: Count authored_by relations for BERT")
    eid = executor.resolve_entity("BERT")
    if eid:
        count = executor.execute_count(eid, "authored_by")
        print(f"  -> {count} authors")

    # Test parsed query
    print("\nTest 3: Parsed query execution")
    query = {
        "type": "1hop",
        "entity": "Attention is All you Need",
        "relations": ["published_in"]
    }
    result = executor.execute_parsed_query(query)
    print(f"  Entity resolved: {result['entity_resolved']}")
    for r in result["results"][:3]:
        print(f"  -> {r['name']}")
