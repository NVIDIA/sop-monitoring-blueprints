# VSS Foundational → VSS SOP Foundational

Modify the upstream `video-search-and-summarization` (branch `3.1.0`) `deployments/foundational/` for the `bp_sop_2d` profile.

**Source:** `../video-search-and-summarization/deployments/foundational/`
**Target:** `../vss-sop/deployments/foundational/`

## Step 0 — Copy from Upstream and Modify for SOP

**Approach:** First find the `foundational` folder in `video-search-and-summarization/deployments/`, copy it to `deployments/foundational/`, then modify the copied folder to work with the SOP profile.

Run these two scripts in order:

1. **Copy from upstream:** `./agentic/vss-sop-skills/vss-sop-build/scripts/copy_foundational_from_upstream.sh`
   - Finds `video-search-and-summarization/deployments/foundational/`
   - Copies entire tree to `deployments/foundational/`

2. **Modify for SOP:** `./agentic/vss-sop-skills/vss-sop-build/scripts/modify_foundational_for_sop.sh`
   - Adds `bp_sop_2d` profile to all services + strips `MINIMAL_PROFILE` suffixes (via `patch_profiles.py`)
   - Replaces ES custom build with stock `docker.elastic.co/elasticsearch/elasticsearch:9.3.0`
   - Removes env blocks (init scripts hardcode values)
   - Renames Kafka topics (`mdx-vlm` removed, `mdx-embed-filtered` → `mdx-vlm-captions`)
   - Tunes health checks (ES retries 15→5, Kibana retries 30→5, start_period 60s→30s)
   - Modifies Logstash configs (adds `mdx-vlm-captions` Kafka input with JSON codec, filter branch, @timestamp preservation)
   - Hardcodes ES init script connection vars (`ELASTICSEARCH_*` → `ES_*`)
   - Removes `BP_PROFILE` conditional branches in ES template creation
   - Adds `mdx_vlm_captions_template` and `mdx-vlm-captions-ilm-policy`
   - Hardcodes `localhost:9092` / `localhost 6379` in broker health-check scripts
   - Removes unused ES Dockerfiles

All steps below describe the individual changes for reference/manual builds.

## Files Modified / Removed

| File | Key Changes |
|---|---|
| `mdx-foundational.yml` | Add `bp_sop_2d` to all profiles; stock ES image; remove ES build, env blocks, `MINIMAL_PROFILE` suffixes; rename Kafka topics; tune health checks |
| `elk/configs/mdx-kafka-logstash.conf` | `mdx-embed-filtered` → `mdx-embed`; **add** `mdx-vlm-captions` Kafka input (JSON codec) + filter branch (`first_timestamp+start_time` → `@timestamp`); preserve `@timestamp` in final mutate |
| `elk/configs/mdx-redis-logstash.conf` | `mdx-embed-filtered` → `mdx-embed` |
| `elk/init-scripts/elasticsearch-ilm-policy-creation.sh` | Hardcode connection vars (`ES_*`); add `mdx-vlm-captions-ilm-policy`; rename `mdx-embed-filtered-ilm-policy` |
| `elk/init-scripts/elasticsearch-ingest-pipeline-creation.sh` | Hardcode `ES_*` vars |
| `elk/init-scripts/elasticsearch-template-creation.sh` | Hardcode `ES_*`; remove `BP_PROFILE` branches; rename `mdx-embed-filtered`; **add** `mdx_vlm_captions_template` for `mdx-vlm-captions-*` |
| `kafka/init-scripts/create-kafka-topics.sh` | Remove parameterized `KAFKA_HOST`/`KAFKA_PORT` |
| `broker-health-check/scripts/check-kafka-health.sh` | Hardcode `localhost:9092` |
| `broker-health-check/scripts/check-redis-health.sh` | Hardcode `localhost:6379` |
| **Removed:** `Dockerfiles/elasticsearch.Dockerfile`, `Dockerfiles/elasticsearch-gpu.Dockerfile` | SOP uses stock `docker.elastic.co/elasticsearch/elasticsearch:9.3.0` |

---

## Step 1 — `mdx-foundational.yml`

### 1a & 1b. Add `bp_sop_2d` to profiles & strip `${MINIMAL_PROFILE:+_extended}` suffixes

You can perform both of these steps automatically and flawlessly using the robust Python patching script:

```bash
python3 agentic/vss-sop-skills/vss-sop-build/scripts/patch_profiles.py deployments/foundational/mdx-foundational.yml
```

This automates:
1. Stripping all `${MINIMAL_PROFILE:+_extended}` suffixes from service profile lists.
2. Robustly prepending `"bp_sop_2d"` as the first entry of the `profiles:` list for all services (including `redis`, `elasticsearch`, `elasticsearch-init-container`, `logstash`, `logstash-redis`, `kafka`, `broker-health-check`, `kibana`, `phoenix`), maintaining perfect YAML syntax, quotes, brackets, and indentation.

*(Manual fallback: For each of these services — `redis`, `elasticsearch`, `elasticsearch-init-container`, `logstash`, `logstash-redis`, `kafka`, `broker-health-check`, `kibana`, `phoenix` — add `"bp_sop_2d"` as the **first entry** of the `profiles:` list (resulting shape: `profiles: ["bp_sop_2d", "bp_wh_2d", ...]`) and strip `${MINIMAL_PROFILE:+_extended}` suffixes).*

### 1c. Replace custom Elasticsearch build with stock image

Replace the entire `build:` block on the `elasticsearch` service with the snippet in `./configs/foundational/elasticsearch-stock-image.yml` — a single `image: docker.elastic.co/elasticsearch/elasticsearch:9.3.0` directive.

### 1d. Remove env blocks (init scripts hardcode values)

- `elasticsearch-init-container.environment` (drop `BP_PROFILE`, `ELASTICSEARCH_*`, `*_DIM` vars)
- `broker-health-check.environment` (drop `BOOTSTRAP_HOST`, `KAFKA_PORT`, `REDIS_PORT`)
- `kafka.environment` (drop `BOOTSTRAP_HOST`, `KAFKA_PORT`)
- `kibana.environment` (drop `SERVER_PUBLICBASEURL`, `SERVER_SECURITYRESPONSEHEADERS_DISABLEEMBEDDING`, `CSP_STRICT`)

### 1e. Rename Kafka topics (in both `kafka` and `broker-health-check` services)

```diff
- KAFKA_TOPICS: '...,{"name": "mdx-vlm"},{"name": "mdx-embed-filtered"}]'
+ KAFKA_TOPICS: '...,{"name": "mdx-vlm-captions"}]'
```

(Removes `mdx-vlm`; renames `mdx-embed-filtered` → `mdx-vlm-captions`.)

> **Important:** `mdx-vlm-captions` MUST appear in the `KAFKA_TOPICS` env var for **both** init containers: the `kafka-topic-init-container` service AND the `broker-health-check` service. If either block is missing the topic, Logstash's Kafka input will connect successfully but never receive any messages (the topic doesn't exist yet when the producer tries to publish). Both blocks must be updated — the diff above applies to each one.

### 1f. Tune health checks + remove `network: host` from build blocks

| Service | Upstream → SOP |
|---|---|
| `elasticsearch` retries | 15 → 5 |
| `kibana` retries | 30 → 5 |
| `kibana` start_period | 60s → 30s |

Remove `network: "host"` from the `build:` blocks of `elasticsearch-init-container` and `broker-health-check`.

---

## Step 2 — Logstash Configs

### 2a. `elk/configs/mdx-kafka-logstash.conf` — three changes

**(1) Rename embed type:** `type => "mdx-embed-filtered"` → `type => "mdx-embed"`

**(2) Add `mdx-vlm-captions` Kafka input** at the end of `input { ... }` (after the `mdx-embed` block):

```
kafka {
    type => "mdx-vlm-captions"
    consumer_threads => 4
    topics => ["mdx-vlm-captions"]
    auto_offset_reset => "earliest"
    decorate_events => true
    group_id => "logstash"
    key_deserializer_class => "org.apache.kafka.common.serialization.StringDeserializer"
    value_deserializer_class => "org.apache.kafka.common.serialization.StringDeserializer"
    codec => "json"
    bootstrap_servers => "localhost:9092"
}
```

Notes (all required):
- `codec => "json"` — DS-SOP runs with `SOP_MESSAGING_SCHEMA=JSON`
- `value_deserializer_class` must be `StringDeserializer` to match
- `auto_offset_reset => "earliest"` so first-boot Logstash drains buffered messages

**(3) Wrap protobuf timestamp ruby + add JSON branch + preserve `@timestamp` in final mutate.** Make the existing protobuf `[timestamp][seconds]` ruby/date conditional on `[type] != "mdx-vlm-captions"`, then add:

```
if [type] == "mdx-vlm-captions" {
    # ds-sop JSON: first_timestamp (Unix seconds, float) + start_time (chunk offset, float)
    ruby {
        code => "event.set('timestamp', ((event.get('first_timestamp').to_f + event.get('start_time').to_f) * 1000).to_i)"
    }
    date {
        match => [ "timestamp","UNIX_MS" ]
        target => "@timestamp"
        timezone => "UTC"
    }
    date {
        match => [ "timestamp","UNIX_MS" ]
        target => "timestamp"
        timezone => "UTC"
    }
}
```

At the **end** of `filter { ... }`, replace the unconditional `mutate { remove_field => ["kafka", "message", "@timestamp", "@version"] }` with a conditional that **keeps `@timestamp`** for `mdx-vlm-captions` (Kibana's data view uses it as `timeFieldName`):

```
if [type] == "mdx-vlm-captions" {
    mutate { remove_field => ["kafka", "message", "@version"] }
} else {
    mutate { remove_field => ["kafka", "message", "@timestamp", "@version"] }
}
```

No output-section change needed: `mdx-vlm-captions` falls through to the existing `else` branch (no `Id`), which writes to `mdx-vlm-captions-<YYYY-MM-DD>` with auto-generated doc IDs.

> **Important — grok double-matching:** The output section uses `[@metadata][year]`, `[@metadata][month]`, `[@metadata][day]` to build the daily index name. The shared `grok` block that sets these three fields must NOT run for `mdx-vlm-captions` events. Because the `date` filter above converts `timestamp` to a Logstash Timestamp object before grok runs, grok may match the same field twice and produce array metadata values, resulting in invalid index names like `mdx-vlm-captions-2026,2026-05,05-28,28` which Elasticsearch rejects with `invalid_index_name_exception`.
>
> **Fix:** wrap the shared grok in `if [type] != "mdx-vlm-captions"` and set the three `[@metadata]` fields directly in the `mdx-vlm-captions` filter branch via a `ruby` block:
>
> ```logstash
> if [type] == "mdx-vlm-captions" {
>     # ... existing date/ruby blocks above ...
>     ruby {
>         code => "
>             t = event.get('@timestamp')
>             ts = t.to_s
>             event.set('[@metadata][year]',  ts[0,4])
>             event.set('[@metadata][month]', ts[5,2])
>             event.set('[@metadata][day]',   ts[8,2])
>         "
>     }
> }
> ```
>
> And the shared grok (which normally sets these fields for other event types) becomes:
>
> ```logstash
> if [type] != "mdx-vlm-captions" {
>     grok {
>         match => { "@timestamp" => "%{YEAR:[@metadata][year]}-%{MONTHNUM:[@metadata][month]}-%{MONTHDAY:[@metadata][day]}" }
>     }
> }
> ```

### 2b. `elk/configs/mdx-redis-logstash.conf`

```diff
- stream_key => "mdx-embed-filtered"
+ stream_key => "mdx-embed"
```

---

## Step 3 — Elasticsearch Init Scripts

All three scripts switch from parameterized `${ELASTICSEARCH_*}` (often defaulted) to **hardcoded `ES_*`** local variables.

### 3a. `elasticsearch-ilm-policy-creation.sh`

| Change | From → To |
|---|---|
| Connection vars | `${ELASTICSEARCH_CONNECTION_MAX_ATTEMPTS:-20}` → hardcoded `ES_CONNECTION_MAX_ATTEMPTS=10` |
| Variable prefix | `ELASTICSEARCH_*` → `ES_*` |
| ILM `min_age` | `${ELASTICSEARCH_ILM_MIN_AGE:-4h}` → hardcoded `4h` |
| `mdx-embed-filtered-ilm-policy` | renamed → `mdx-embed-ilm-policy` |
| `mdx-vlm-captions-ilm-policy` | **added** |

### 3b. `elasticsearch-ingest-pipeline-creation.sh`

Same connection-var changes as 3a (parameterized → hardcoded `ES_*`).

### 3c. `elasticsearch-template-creation.sh`

| Change | From → To |
|---|---|
| Connection vars | parameterized → hardcoded `ES_*` |
| `BP_PROFILE` conditional | `if [[ "${BP_PROFILE}" == "bp_developer_search" ]] ...` (behaviour + raw `dense_vector` templates) → removed (single template variant) |
| Embedding dims | `${ELASTICSEARCH_*_DIM}` → hardcoded `768` |
| `mdx_embed_filtered_template` | renamed → `mdx_embed_template`, `index_patterns: ["mdx-embed-*"]` |
| `mdx_vlm_captions_template` | **added** (see below) |

**New `mdx_vlm_captions_template`:**

```json
{
  "index_patterns": ["mdx-vlm-captions-*"],
  "priority": 516,
  "template": {
    "settings": { "index.lifecycle.name": "mdx-vlm-captions-ilm-policy" },
    "mappings": {
      "dynamic": true,
      "properties": {
        "cv_execute_time":  { "type": "double" },
        "vlm_execute_time": { "type": "double" },
        "frame_number":     { "type": "integer" },
        "chunk_idx":        { "type": "integer" }
      }
    }
  }
}
```

---

## Step 4 — Kafka & Health-Check Scripts

| Script | Change |
|---|---|
| `kafka/init-scripts/create-kafka-topics.sh` | Remove `KAFKA_HOST=${BOOTSTRAP_HOST:-localhost}` and `KAFKA_PORT=${KAFKA_PORT:-9092}` (script uses `localhost:9092` directly) |
| `broker-health-check/scripts/check-kafka-health.sh` | Replace all `$KAFKA_HOST:$KAFKA_PORT` with hardcoded `localhost:9092`; drop the variable declarations and their echo statements |
| `broker-health-check/scripts/check-redis-health.sh` | Replace all `$REDIS_HOST $REDIS_PORT` with hardcoded `localhost 6379`; drop the variable declarations and echo statements |

---

## Step 5 — Remove Unused Dockerfiles

SOP uses the stock ES image, so the custom Elasticsearch Dockerfiles are removed:

```bash
rm -f deployments/foundational/Dockerfiles/elasticsearch.Dockerfile
rm -f deployments/foundational/Dockerfiles/elasticsearch-gpu.Dockerfile
```

This is performed automatically as Step 3 of `modify_foundational_for_sop.sh`.

---

## Verification

Run `./scripts/foundational/verify.sh` (a thin wrapper that delegates to `scripts/verify_build.py --component foundational`, the single source of truth) to validate:

1. `bp_sop_2d` coverage in `mdx-foundational.yml`
2. No `MINIMAL_PROFILE` references remain
3. Stock Elasticsearch image is used
4. `mdx-vlm-captions` Kafka topic present; no `embed-filtered` anywhere
5. Custom ES Dockerfiles removed
6. `mdx-vlm-captions` Kafka input with JSON codec in Logstash
7. Filter wires `timestamp` from `first_timestamp + start_time`
8. Final `mutate` preserves `@timestamp` for `mdx-vlm-captions`

A FAIL on check 6 means `vss-sop-test` will report 0 ES indices; a FAIL on check 8 means `vss-sop-test kibana_dashboard_fields` will fail because the data view's `timeFieldName` (`@timestamp`) is missing from the ES mapping.

---

## Troubleshooting

- **Logstash `ConfigurationError: Expected one of [ \t\r\n], "#", "}"` at a `date` block** — Logstash DSL does **not** allow inline plugin options separated by semicolons (e.g. `date { match => [...]; target => ...; timezone => ... }`). Each option must be on its own line. Use the multi-line form:
  ```logstash
  date {
      match => ["timestamp", "UNIX_MS"]
      target => "@timestamp"
      timezone => "UTC"
  }
  ```
  This applies to every `date { }` block in `mdx-kafka-logstash.conf`.

- **`mdx-vlm-captions` index name contains commas (e.g. `mdx-vlm-captions-2026,2026-05,05-28,28`) → ES rejects with `invalid_index_name_exception`** — The shared `grok` block that sets `[@metadata][year/month/day]` ran for VLM captions events and double-populated the metadata fields as arrays (because `timestamp` is already a Logstash Timestamp object at that point, not a plain string). Fix: wrap the grok in `if [type] != "mdx-vlm-captions"` and set the three `[@metadata]` fields directly in the VLM captions filter branch via a `ruby` block extracting substrings from `@timestamp.to_s` (see Step 2a item 3 above). Also ensure `mdx-vlm-captions` is listed in both `KAFKA_TOPICS` env var blocks in `mdx-foundational.yml` (Step 1e).

- **Logstash connects to Kafka but `mdx-vlm-captions` index never appears** — Verify that (a) `mdx-vlm-captions` is in `KAFKA_TOPICS` for both the `kafka-topic-init-container` and `broker-health-check` services, (b) DS-SOP has `SOP_MESSAGING_SCHEMA=JSON` and `ENABLE_MESSAGING=1`, and (c) Logstash `mdx-kafka-logstash.conf` has the `mdx-vlm-captions` input block with `codec => "json"`. A missing topic in `KAFKA_TOPICS` means the topic is never created, so DS-SOP's producer gets `UnknownTopicOrPartitionException` and no messages are delivered.

- **ES `mdx-vlm-captions` doc count stuck at 1** — `mdx-vlm-captions` was accidentally added to the Logstash output `if` branch that uses `document_id => "%{Id}"`. Because no fingerprint block sets `Id` for that type, every document overwrites the same blank-ID entry. Fix: remove `or [type] == "mdx-vlm-captions"` from the output `if` condition so it falls through to the `else` branch (auto-generated doc IDs). The `foundational` component of `verify_build.py` detects this.

- **Kibana "No field found for [llm.queries.response.keyword]"** — the `sop-kibana-objects.ndjson` data view must use **flat JSON** field names (DS-SOP `SOP_MESSAGING_SCHEMA=JSON` output), not nested protobuf paths. Use `response.keyword`, `sensor_id.keyword`, `cv_execute_time`, `vlm_execute_time`, `chunk_idx`, `frame_number`, `@timestamp` (time field), `start_time`/`end_time` floats. Validate with `curl localhost:9200/mdx-vlm-captions-*/_mapping?pretty`.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
