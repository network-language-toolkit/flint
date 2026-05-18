// Cloudflare Worker — proxies the HuggingFace Inference Providers API so the
// static page at flint.networklanguagetoolkit.com/v2/ can request BGE-large
// embeddings without exposing the HF token in client-side code.
//
// Endpoint: POST /embed { "query": "<text>" }   → { "embedding": [<1024 floats>] }
// Errors:   non-200 with { "error": "...", "status": <int>, "body": "..." }
//
// Secrets (set via `wrangler secret put HF_TOKEN`):
//   HF_TOKEN       — the HF API token with inference.serverless.write
// Vars (in wrangler.toml [vars]):
//   BILL_TO        — org name for X-HF-Bill-To header (empty string disables)
//   ALLOWED_ORIGIN — CORS origin (e.g. https://flint.networklanguagetoolkit.com)

const HF_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5/pipeline/feature-extraction";
const MAX_QUERY_LEN = 1000;

export default {
  async fetch(request, env) {
    // ALLOWED_ORIGIN is a comma-separated allowlist. We echo whichever entry
    // matches the request's Origin header — '*' in the list disables the check.
    const allowlist = (env.ALLOWED_ORIGIN || "*")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const requestOrigin = request.headers.get("Origin") || "";
    let origin;
    if (allowlist.includes("*")) {
      origin = "*";
    } else if (allowlist.includes(requestOrigin)) {
      origin = requestOrigin;
    } else {
      // No matching origin — return a CORS-less error response. The browser
      // will block it client-side, but the body still helps with curl debug.
      return new Response(JSON.stringify({ error: "origin_not_allowed", origin: requestOrigin }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders(origin) });
    }

    // Per-IP rate limit (60 req/min). Placed after the origin allowlist
    // and OPTIONS short-circuit so cross-origin abuse gets the cheaper
    // 403 and preflights don't count against the bucket.
    const clientIp = request.headers.get("CF-Connecting-IP") || "unknown";
    const { success } = await env.RATELIMITER.limit({ key: clientIp });
    if (!success) {
      return new Response(
        JSON.stringify({ error: "rate_limited", retry_after: 60 }),
        {
          status: 429,
          headers: {
            "Content-Type": "application/json",
            "Retry-After": "60",
            ...corsHeaders(origin),
          },
        },
      );
    }

    if (request.method !== "POST") {
      return jsonResponse({ error: "method_not_allowed" }, 405, origin);
    }
    if (new URL(request.url).pathname !== "/embed") {
      return jsonResponse({ error: "not_found" }, 404, origin);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return jsonResponse({ error: "invalid_json" }, 400, origin);
    }
    const query = body && body.query;
    if (typeof query !== "string" || query.length === 0 || query.length > MAX_QUERY_LEN) {
      return jsonResponse({ error: "invalid_query" }, 400, origin);
    }

    const hfHeaders = {
      Authorization: `Bearer ${env.HF_TOKEN}`,
      "Content-Type": "application/json",
    };
    if (env.BILL_TO) {
      hfHeaders["X-HF-Bill-To"] = env.BILL_TO;
    }

    let upstream;
    try {
      upstream = await fetch(HF_URL, {
        method: "POST",
        headers: hfHeaders,
        body: JSON.stringify({ inputs: query }),
      });
    } catch (e) {
      return jsonResponse({ error: "upstream_network", detail: String(e) }, 502, origin);
    }

    if (upstream.status !== 200) {
      const text = await upstream.text();
      return jsonResponse(
        { error: "upstream_status", status: upstream.status, body: text.slice(0, 500) },
        upstream.status >= 500 ? 502 : upstream.status,
        origin,
      );
    }

    const embedding = await upstream.json();
    return jsonResponse({ embedding }, 200, origin);
  },
};

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function jsonResponse(payload, status, origin) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(origin),
    },
  });
}
