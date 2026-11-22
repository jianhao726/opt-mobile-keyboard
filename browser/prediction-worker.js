import { Tokenizer } from "@huggingface/tokenizers";
import { env, InferenceSession, Tensor } from "onnxruntime-web/wasm";

const assets = new URL("./assets/", import.meta.url);
env.wasm.numThreads = 1;
env.wasm.wasmPaths = assets.href;

const BLOCKLIST = new Set(["damn", "hell", "shit", "fuck", "bitch", "bastard", "asshole", "crap", "idiot"]);
const MAX_CONTEXT = 128;
const ALPHABET = "abcdefghijklmnopqrstuvwxyz";
let enginePromise;

async function asset(name) {
  const response = await fetch(new URL(name, assets));
  if (!response.ok) throw new Error(`Could not download ${name} (HTTP ${response.status}).`);
  return response;
}

async function loadEngine() {
  const [spec, config, dictionary, model] = await Promise.all([
    asset("tokenizer.json").then(response => response.json()),
    asset("tokenizer_config.json").then(response => response.json()),
    asset("spelling.json").then(response => response.json()),
    asset("opt-int8.onnx").then(response => response.arrayBuffer()),
  ]);
  self.postMessage({ status: "Starting prediction model…" });
  const session = await InferenceSession.create(model, { executionProviders: ["wasm"] });
  const tokenizer = new Tokenizer(spec, config);
  const words = [];
  for (const [token, id] of tokenizer.get_vocab()) {
    const word = token.slice(1);
    const lower = word.toLowerCase();
    if (token.startsWith("Ġ") && /^\p{L}+$/u.test(word) && !BLOCKLIST.has(lower)) {
      words.push({ word, id, lower });
    }
  }
  const punctuation = [".", ",", "?", "!"].map(word => ({ word, id: tokenizer.token_to_id(word) }));
  return new Engine(session, tokenizer, words, punctuation, dictionary);
}

const capitalize = word => word[0].toUpperCase() + word.slice(1).toLowerCase();
const int64 = (values, dimensions) => new Tensor("int64", BigInt64Array.from(values, BigInt), dimensions);

function rank(candidates, score, limit) {
  const ranked = candidates.map(({ word, id, lower }) => ({ word, score: score(id, lower) }))
    .sort((a, b) => b.score - a.score).slice(0, limit);
  if (!ranked.length) return [];
  const weights = ranked.map(entry => Math.exp(entry.score - ranked[0].score));
  const sum = weights.reduce((a, b) => a + b, 0);
  return ranked.map(({ word }, i) => ({ word, probability: weights[i] / sum }));
}

function spellingCandidates(word, dictionary) {
  const lower = word.toLowerCase();
  if (lower.length < 3 || lower.length > 24 || Object.hasOwn(dictionary, lower)) return [];
  const edits = new Set();
  for (let i = 0; i <= lower.length; i++) {
    const left = lower.slice(0, i);
    const right = lower.slice(i);
    if (right.length) edits.add(left + right.slice(1));
    if (right.length > 1) edits.add(left + right[1] + right[0] + right.slice(2));
    for (const letter of ALPHABET) {
      edits.add(left + letter + right);
      if (right.length) edits.add(left + letter + right.slice(1));
    }
  }
  return [...edits].filter(edit => Object.hasOwn(dictionary, edit) && !BLOCKLIST.has(edit))
    .sort((a, b) => dictionary[b] - dictionary[a] || a.localeCompare(b)).slice(0, 5)
    .map(edit => word === word.toUpperCase() ? edit.toUpperCase() : /^[A-Z][a-z]+$/.test(word) ? capitalize(edit) : edit);
}

class Engine {
  constructor(session, tokenizer, words, punctuation, dictionary) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.words = words;
    this.punctuation = punctuation;
    this.dictionary = dictionary;
    this.logitCache = new Map();
    this.spellingCache = new Map();
  }

  encode(text, addSpecialTokens) {
    return this.tokenizer.encode(text, { add_special_tokens: addSpecialTokens }).ids;
  }

  async infer(sequences, positions) {
    const length = Math.max(...sequences.map(sequence => sequence.length));
    const ids = sequences.flatMap(sequence => [...sequence, ...Array(length - sequence.length).fill(1)]);
    const masks = sequences.flatMap(sequence => [...Array(sequence.length).fill(1), ...Array(length - sequence.length).fill(0)]);
    const shape = [sequences.length, length];
    const { logits } = await this.session.run({
      ids: int64(ids, shape), mask: int64(masks, shape), positions: int64(positions, [positions.length]),
    });
    return logits;
  }

  async nextLogits(prefix) {
    const text = prefix.trim() ? prefix : "</s>";
    if (!this.logitCache.has(text)) {
      const ids = this.encode(text, true).slice(0, MAX_CONTEXT);
      const logits = await this.infer([ids], [ids.length - 1]);
      if (this.logitCache.size === 12) this.logitCache.delete(this.logitCache.keys().next().value);
      this.logitCache.set(text, logits.data);
    }
    return this.logitCache.get(text);
  }

  async wordCorrections(word, prefix) {
    const key = JSON.stringify([word, prefix]);
    if (this.spellingCache.has(key)) return this.spellingCache.get(key);
    const candidates = spellingCandidates(word, this.dictionary);
    if (candidates.length < 2) return candidates;
    const left = prefix.trimEnd();
    const leftIds = left ? this.encode(left, false) : [2];
    const alternatives = candidates.map(candidate => this.encode((left ? " " : "") + candidate, false));
    const longest = Math.max(...alternatives.map(ids => ids.length));
    const context = leftIds.slice(-(MAX_CONTEXT - longest));
    const positions = Array.from({ length: longest }, (_, i) => context.length - 1 + i);
    const { data, dims } = await this.infer(alternatives.map(ids => [...context, ...ids]), positions);
    const vocabSize = dims[2];
    const scores = alternatives.map((ids, i) => {
      let score = 0;
      for (const [j, id] of ids.entries()) {
        const start = (i * longest + j) * vocabSize;
        let max = -Infinity;
        for (let k = start; k < start + vocabSize; k++) max = Math.max(max, data[k]);
        let total = 0;
        for (let k = start; k < start + vocabSize; k++) total += Math.exp(data[k] - max);
        score += data[start + id] - max - Math.log(total);
      }
      return { word: candidates[i], score: score / ids.length };
    });
    const ranked = scores.sort((a, b) => b.score - a.score).slice(0, 3).map(item => item.word);
    if (this.spellingCache.size === 64) this.spellingCache.delete(this.spellingCache.keys().next().value);
    this.spellingCache.set(key, ranked);
    return ranked;
  }

  async predict({ draft, context, spelling, cacheWeight, cache }) {
    const startTime = performance.now();
    const previous = context.map(message => message.trim()).filter(Boolean).join(" ");
    const text = previous ? previous + " " + (draft.trim() ? draft : "") : draft;
    const next = text.endsWith(" ");
    const split = text.lastIndexOf(" ");
    const prefix = next ? text.trimEnd() : split >= 0 ? text.slice(0, split) : "";
    const partial = next ? "" : text.slice(split + 1);
    const lower = partial.toLowerCase();
    const mode = !text ? "Starter" : next ? "Next-Word" : "Prefix-Completion";
    let suggestions;
    if (!text) suggestions = [{ word: "I", probability: .32 }, { word: "Hi", probability: .25 }, { word: "Hey", probability: .18 }];
    else {
      const matches = next ? this.words
        : this.words.filter(item => item.lower.startsWith(lower) && item.word.length > partial.length);
      if (matches.length) {
        const logits = await this.nextLogits(prefix);
        suggestions = rank(matches, (id, word) => logits[id] + cacheWeight * Math.log1p(Object.hasOwn(cache, word) ? cache[word] : 0), 3);
      } else {
        suggestions = Object.entries(cache).filter(([word]) => word.startsWith(lower) && word.length > lower.length)
          .sort((a, b) => b[1] - a[1]).slice(0, 3).map(([word]) => ({ word, probability: .9 }));
      }
      if (/^\p{Lu}/u.test(partial)) suggestions.forEach(item => { item.word = capitalize(item.word); });
    }
    let punctuation = null;
    if (draft.trim() && /[\p{L}\p{N}]$/u.test(text.trimEnd()) && (next || text.trimEnd().includes(" "))) {
      const logits = await this.nextLogits(text.trimEnd());
      punctuation = rank(this.punctuation, id => logits[id], 4)[0];
    }
    const corrections = [];
    if (spelling && draft) {
      for (const match of draft.matchAll(/(?<![\p{L}\p{N}_'’@/.-])([A-Za-z]+)(?=[,.!?;:]*\s)/gu)) {
        if (Object.hasOwn(cache, match[1].toLowerCase())) continue;
        const left = draft.slice(0, match.index);
        const candidates = await this.wordCorrections(match[1], (previous ? previous + " " : "") + left);
        if (candidates.length) {
          const start = Array.from(left).length;
          corrections.push({ word: match[1], start, end: start + match[1].length, candidates });
        }
      }
    }
    return { suggestions, punctuation, corrections, mode, latencyMs: performance.now() - startTime };
  }
}

self.onmessage = async ({ data: request }) => {
  try {
    enginePromise ||= loadEngine().catch(error => {
      enginePromise = null;
      throw error;
    });
    const engine = await enginePromise;
    self.postMessage({ result: await engine.predict(request) });
  } catch (error) {
    console.error("Browser prediction failed", error);
    self.postMessage({ error: "The browser prediction engine could not run. Retry or try a recent Chrome, Edge, or Firefox." });
  }
};
