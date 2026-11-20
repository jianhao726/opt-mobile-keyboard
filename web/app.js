(() => {
  "use strict";

  function init() {
    const el = (id) => document.getElementById(id);
    const input = el("draft");
    const examples = location.protocol === "file:" || document.body.dataset.preview === "true";
    const soloPrompts = [
      ["Low battery", "My phone battery is at 1%, do you have a "],
      ["Train meetup", "Sure, I will meet you at the train "],
      ["Grocery run", "I am heading to the groc"],
      ["Lunch plans", "Hey! Are you free for lun"],
    ];
    const chatPrompts = [
      ["Lunch invite", "Hey! Are you free for lunch today?", ""],
      ["Phone battery", "My phone battery is at 1%, do you have a charger?", "Yes, I have a "],
      ["Train meetup", "I just arrived at the station, where are you?", "I am on my "],
    ];
    const profiles = Object.fromEntries(["Solo", "User A", "User B"].map((name) => [
      name, { draft: "", cache: new Map(), dismissed: new Set() },
    ]));
    const logs = { solo: [], chat: [] };
    let mode = "solo";
    let shifted = false;
    let numbers = false;
    let predictionVersion = 0;
    let timer;
    let busy = false;
    let queued = false;
    let modelHasResponded = false;

    const profileName = () => mode === "solo" ? "Solo" : el("speaker").value;
    const profile = () => profiles[profileName()];
    const make = (tag, className, value) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (value !== undefined) node.textContent = value;
      return node;
    };
    const button = (className, label) => {
      const node = make("button", className, label);
      node.type = "button";
      return node;
    };
    const focusDraft = (offset = input.value.length) => {
      input.focus({ preventScroll: true });
      input.setSelectionRange(offset, offset);
    };
    const status = (value, state) => {
      el("engine-status").textContent = value;
      el("engine-status").dataset.state = state;
    };
    const signature = (word, start) => JSON.stringify([word.toLowerCase(), start]);

    function setTab(tab) {
      for (const name of ["compose", "history"]) {
        const selected = name === tab;
        const trigger = el("tab-" + name);
        trigger.setAttribute("aria-selected", String(selected));
        trigger.tabIndex = selected ? 0 : -1;
        el("view-" + name).hidden = !selected;
      }
    }

    function setDraft(value, caret) {
      input.value = value;
      profile().draft = value;
      el("send").disabled = !value.trim();
      if (mode === "chat") {
        el("reset-chat").disabled = !logs.chat.length && !profiles["User A"].draft && !profiles["User B"].draft;
      }
      if (!value) profile().dismissed.clear();
      if (caret !== undefined) focusDraft(caret);
      schedulePrediction();
    }

    function learn(text) {
      const cache = profile().cache;
      for (const token of text.split(/\s+/u)) {
        const word = token.replace(/^[.,!?;:'"]+|[.,!?;:'"]+$/g, "").toLowerCase();
        if (!/^\p{L}{2,64}$/u.test(word)) continue;
        if (cache.has(word) || cache.size < 1024) {
          cache.set(word, Math.min(1_000_000, (cache.get(word) || 0) + 1));
        }
      }
      renderVocabulary();
    }

    function renderVocabulary() {
      const words = [...profile().cache].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      el("reset-cache").disabled = words.length === 0;
      el("profile-label").textContent = profileName();
      const items = words.map(([word, count]) => {
        const item = make("li", "vocab-item");
        item.append(make("span", "vocab-word", word), make("span", "vocab-count", count));
        return item;
      });
      if (!items.length) items.push(make("li", "vocab-empty", "Send a message or choose a suggestion to add words."));
      el("vocabulary").replaceChildren(...items);
    }

    function renderHistory() {
      const messages = logs[mode];
      el("history-count").textContent = messages.length + (messages.length === 1 ? " message" : " messages");
      el("history-empty").hidden = messages.length > 0;
      const recent = el("chat-recent");
      recent.hidden = mode !== "chat" || !messages.length;
      recent.replaceChildren(...(recent.hidden ? [] : messages.slice(-2).map(({ sender, text }) => {
        const row = make("div", "recent-turn");
        row.append(make("span", "", sender), make("span", "", text));
        return row;
      })));
      const rows = messages.map(({ sender, text }) => {
        const row = make("div", "history-item");
        row.dataset.speaker = sender;
        row.append(make("span", "history-sender", sender === "Solo" ? "You" : sender),
          make("p", "history-text", text));
        return row;
      });
      el("history").replaceChildren(...rows);
    }

    function send() {
      const text = input.value.trim();
      if (!text) return;
      const sender = profileName();
      logs[mode].push({ sender, text });
      learn(text);
      profile().draft = "";
      profile().dismissed.clear();
      if (mode === "chat" && el("auto-switch").checked) {
        el("speaker").value = sender === "User A" ? "User B" : "User A";
      }
      setDraft(profile().draft, profile().draft.length);
      el("send-status").textContent = mode === "chat" ? "Sent as " + sender + "." : "Message sent.";
      renderHistory();
      renderVocabulary();
    }

    function loadPrompt(entry) {
      if (mode === "chat") {
        logs.chat = [{ sender: "User A", text: entry[1] }];
        for (const name of ["User A", "User B"]) {
          profiles[name].draft = "";
          profiles[name].dismissed.clear();
        }
        el("speaker").value = "User B";
        renderHistory();
      }
      const value = entry[mode === "solo" ? 1 : 2];
      profile().dismissed.clear();
      el("send-status").textContent = mode === "chat" ? "Conversation loaded. User B is typing." : "";
      setTab("compose");
      setDraft(value, value.length);
      renderVocabulary();
    }

    function renderMode() {
      el("mode-solo").setAttribute("aria-pressed", String(mode === "solo"));
      el("mode-chat").setAttribute("aria-pressed", String(mode === "chat"));
      el("chat-controls").hidden = mode !== "chat";
      el("send-status").textContent = "";
      el("prompt-intro").textContent = mode === "solo" ? "Pick a phrase to start typing." : "Start a conversation and try a reply.";
      const entries = mode === "solo" ? soloPrompts : chatPrompts;
      el("prompts").replaceChildren(...entries.map((entry) => {
        const node = button("prompt");
        const arrow = make("span", "prompt-arrow", "↗");
        arrow.setAttribute("aria-hidden", "true");
        node.append(make("span", "prompt-name", entry[0]), make("span", "prompt-preview", entry[1].trimEnd() + (mode === "solo" ? "…" : "")), arrow);
        node.addEventListener("click", () => loadPrompt(entry));
        return node;
      }));
      setDraft(profile().draft);
      renderHistory();
      renderVocabulary();
    }

    // Examples demonstrate editing when the HTML opens outside the Python server.
    // Confidence and engine metrics are available only from the real model.
    function examplePrediction(draft, context) {
      const sentence = [...context, draft].join(" ");
      const words = ["I", "Hi", "Hey", "grocery", "groceries", "grocer", "lunch", "lunchtime",
        "lunchbox", "train", "station", "platform", "stop", "charger", "cable", "power",
        "way", "waypoint", "wayward", "sure", "yes", "sounds", "great", "thanks", "tomorrow",
        "today", "meeting", "meet", "there", "home", "hello", "help", "happy", "you", "your"];
      let wordList;
      let predictionMode;
      if (!sentence.trim()) {
        predictionMode = "Starter";
        wordList = words.slice(0, 3);
      } else if (/\s$/.test(sentence)) {
        predictionMode = "Next-Word";
        const choices = [
          [/train\s*$/i, ["station", "platform", "stop"]],
          [/have a\s*$/i, ["charger", "cable", "minute"]],
          [/on my\s*$/i, ["way", "phone", "break"]],
          [/free for\s*$/i, ["lunch", "dinner", "coffee"]],
          [/lunch today\?\s*$/i, ["Yes", "Sure", "Sorry"]],
          [/\b(I|we)\s*$/i, ["will", "can", "have"]],
        ];
        wordList = choices.find(([pattern]) => pattern.test(sentence))?.[1] || ["the", "a", "to"];
      } else {
        predictionMode = "Prefix-Completion";
        const prefix = draft.split(/\s/).pop() || "";
        wordList = words.filter((word) => word.length > prefix.length && word.toLowerCase().startsWith(prefix.toLowerCase())).slice(0, 3);
        if (/^[A-Z]/.test(prefix)) wordList = wordList.map((word) => word[0].toUpperCase() + word.slice(1));
      }
      return {
        suggestions: wordList.map((word) => ({ word })), mode: predictionMode,
        punctuation: /[\p{L}\p{N}]\s*$/u.test(draft) && /\s/.test(draft) ? { word: "." } : null,
        corrections: [],
      };
    }

    function chooseWord(word, predictionMode, punctuation) {
      const current = input.value;
      let result;
      if (punctuation) result = current.trimEnd() + word + " ";
      else if (predictionMode === "Prefix-Completion") result = current.replace(/[^\s]*$/u, () => word + " ");
      else result = current + (current && !/\s$/.test(current) ? " " : "") + word + " ";
      if (result.length > input.maxLength) return;
      learn(word);
      setDraft(result, result.length);
    }

    function showCorrections(corrections) {
      const draft = input.value;
      const letters = Array.from(draft);
      const rows = [];
      for (const { word, start, end, candidates } of corrections) {
        const key = signature(word, start);
        if (profile().dismissed.has(key) || letters.slice(start, end).join("") !== word) continue;
        const row = make("div", "correction");
        row.append(make("span", "correction-label", "Check “" + word + "”"));
        for (const replacement of candidates) {
          const use = button("correction-choice", replacement);
          use.setAttribute("aria-label", "Replace " + word + " with " + replacement);
          use.addEventListener("click", () => {
            const before = letters.slice(0, start).join("");
            const corrected = before + replacement + letters.slice(end).join("");
            if (input.value !== draft || corrected.length > input.maxLength) return;
            const delta = Array.from(replacement).length - (end - start);
            const shifted = new Set();
            for (const prior of profile().dismissed) {
              const [name, position] = JSON.parse(prior);
              if (position !== start) shifted.add(signature(name, position > start ? position + delta : position));
            }
            profile().dismissed = shifted;
            learn(replacement);
            setDraft(corrected, before.length + replacement.length);
          });
          row.append(use);
        }
        const keep = button("correction-keep", "Keep original");
        keep.addEventListener("click", () => {
          profile().dismissed.add(key);
          row.remove();
          focusDraft();
        });
        row.append(keep);
        rows.push(row);
      }
      el("corrections").replaceChildren(...rows);
    }

    function showPrediction(result) {
      status(examples ? "Example preview" : "Model ready", examples ? "preview" : "ready");
      el("latency").textContent = examples ? "—" : Number(result.latencyMs).toFixed(1) + " ms";
      el("footprint").textContent = examples ? "—" : result.footprint;
      const modeLabels = { "Starter": "Starter", "Next-Word": "Next word", "Prefix-Completion": "Completion" };
      el("prediction-mode").textContent = modeLabels[result.mode] || result.mode;
      const choices = [...result.suggestions];
      if (result.punctuation) choices.push(result.punctuation);
      const buttons = choices.map(({ word, probability }, i) => {
        const punct = i === result.suggestions.length;
        const node = button("suggestion" + (punct ? " suggestion-punct" : ""));
        node.append(make("span", "suggest-word", word));
        const prob = make("span", "suggest-prob", Number.isFinite(probability) ? Math.round(probability * 100) + "%" : "");
        prob.hidden = !Number.isFinite(probability);
        node.append(prob);
        node.setAttribute("aria-label", (punct ? "Add punctuation " : "Use suggestion ") + word);
        node.addEventListener("click", () => chooseWord(word, result.mode, punct));
        return node;
      });
      if (!buttons.length) buttons.push(make("span", "suggest-empty", "No matches. Keep typing."));
      el("suggestions").replaceChildren(...buttons);
      showCorrections(result.corrections);
    }

    function showError(message) {
      status("Engine unavailable", "error");
      for (const id of ["latency", "footprint", "prediction-mode"]) el(id).textContent = "—";
      const retry = button("suggestion retry", "Retry");
      retry.addEventListener("click", () => schedulePrediction(0));
      el("suggestions").replaceChildren(retry);
      el("corrections").replaceChildren(make("span", "engine-error", message));
    }

    async function predict(version) {
      if (busy) {
        queued = true;
        return;
      }
      busy = true;
      const context = mode === "chat" ? logs.chat.slice(-2).map((message) => message.text) : [];
      try {
        let result;
        if (examples) result = examplePrediction(input.value, context);
        else {
          const request = { draft: input.value, context, variant: el("variant").value,
            cacheWeight: Number(el("boost").value), spelling: el("spelling").checked, cache: Object.create(null) };
          const utf8 = new TextEncoder();
          let bytes = utf8.encode(JSON.stringify(request)).length;
          for (const [word, count] of [...profile().cache].sort((a, b) => b[1] - a[1])) {
            const entryBytes = utf8.encode(JSON.stringify(word) + ":" + count + ",").length;
            if (bytes + entryBytes <= 128 * 1024) {
              request.cache[word] = count;
              bytes += entryBytes;
            }
          }
          const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request),
          });
          if (!response.headers.get("content-type")?.includes("application/json")) {
            throw new Error("Start web_app.py to connect the prediction engine.");
          }
          result = await response.json();
          if (!response.ok) throw new Error(result.error || "The prediction engine could not complete this request.");
          modelHasResponded = true;
        }
        if (version === predictionVersion) showPrediction(result);
      } catch (error) {
        if (version === predictionVersion) showError(error instanceof TypeError ? "Could not reach the prediction server." : error.message);
      } finally {
        if (version === predictionVersion) el("suggestions").setAttribute("aria-busy", "false");
        busy = false;
        if (queued) {
          queued = false;
          clearTimeout(timer);
          void predict(predictionVersion);
        }
      }
    }

    function schedulePrediction(delay = 160) {
      const version = ++predictionVersion;
      clearTimeout(timer);
      queued = false;
      for (const node of el("suggestions").querySelectorAll("button")) node.disabled = true;
      el("suggestions").setAttribute("aria-busy", "true");
      el("corrections").replaceChildren();
      if (!examples) status(modelHasResponded ? "Predicting…" : "Loading engine…", "loading");
      if (examples) void predict(version);
      else timer = setTimeout(() => void predict(version), delay);
    }

    function drawKeyboard() {
      const rows = numbers ? [
        [..."1234567890"], [..."@#$%&*-+()"], ["!", "?", "'", '"', "/", ":", ";", "Backspace"],
      ] : [[..."qwertyuiop"], [..."asdfghjkl"], ["Shift", ..."zxcvbnm", "Backspace"]];
      rows.push(["123", ",", "Space", ".", "Enter"]);
      const labels = { Shift: "⇧", Backspace: "⌫", Space: "space", Enter: "↵", "123": numbers ? "ABC" : "123" };
      const names = { Shift: "Shift", Backspace: "Backspace", Space: "Space", Enter: "Send message", "123": numbers ? "Show letters" : "Show numbers and symbols" };
      el("keyboard").replaceChildren(...rows.map((keys) => {
        const row = make("div", "key-row");
        for (const key of keys) {
          const node = button("key" + (names[key] ? " key-fn" : "") + (key === "Space" ? " key-space" : ""), labels[key] || (shifted ? key.toUpperCase() : key));
          node.dataset.key = key;
          node.setAttribute("aria-label", names[key] || node.textContent);
          if (key === "Shift") node.setAttribute("aria-pressed", String(shifted));
          row.append(node);
        }
        return row;
      }));
    }

    function pressKey(key) {
      if (key === "Shift" || key === "123") {
        if (key === "Shift") shifted = !shifted;
        else { numbers = !numbers; shifted = false; }
        drawKeyboard();
        return;
      }
      if (key === "Enter") return send();
      let start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? start;
      let insert = key === "Space" ? " " : shifted ? key.toUpperCase() : key;
      if (key === "Backspace") {
        insert = "";
        if (start === end && start) start -= Array.from(input.value.slice(0, start)).pop().length;
      }
      if (input.value.length - (end - start) + insert.length > input.maxLength) return;
      input.setRangeText(insert, start, end, "end");
      if (shifted && insert) { shifted = false; drawKeyboard(); }
      setDraft(input.value);
      input.setSelectionRange(start + insert.length, start + insert.length);
    }

    const settings = el("settings-dialog");
    el("open-settings").addEventListener("click", () => settings.showModal());
    el("close-settings").addEventListener("click", () => settings.close());
    settings.addEventListener("click", (event) => {
      if (event.target !== settings) return;
      const bounds = settings.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) settings.close();
    });
    for (const tab of ["compose", "history"]) {
      const node = el("tab-" + tab);
      node.addEventListener("click", () => setTab(tab));
      node.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? "compose" : event.key === "End" ? "history" : tab === "compose" ? "history" : "compose";
        setTab(next);
        el("tab-" + next).focus();
      });
    }
    for (const next of ["solo", "chat"]) {
      el("mode-" + next).addEventListener("click", () => { mode = next; renderMode(); });
    }
    el("speaker").addEventListener("change", renderMode);
    el("reset-chat").addEventListener("click", () => {
      logs.chat = [];
      for (const name of ["User A", "User B"]) {
        profiles[name].draft = "";
        profiles[name].dismissed.clear();
      }
      el("speaker").value = "User A";
      renderMode();
    });
    el("reset-cache").addEventListener("click", () => { profile().cache.clear(); renderVocabulary(); schedulePrediction(); });
    input.maxLength = 2048;
    input.addEventListener("input", () => {
      setDraft(input.value);
      el("send-status").textContent = "";
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); send(); }
    });
    el("send").addEventListener("click", send);
    el("keyboard").addEventListener("pointerdown", (event) => { if (event.target.closest("[data-key]")) event.preventDefault(); });
    el("keyboard").addEventListener("click", (event) => {
      const key = event.target.closest("button[data-key]");
      if (key) pressKey(key.dataset.key);
    });
    el("boost").addEventListener("input", () => {
      el("boost-value").textContent = Number(el("boost").value).toFixed(2);
      schedulePrediction();
    });
    for (const id of ["variant", "spelling"]) el(id).addEventListener("change", () => schedulePrediction());
    if (examples) {
      document.querySelector(".engine-name").textContent = "Word list";
      const explanation = make("small", "preview-note", "This preview uses example words. Start the Python app to use OPT-125M and adjust its settings.");
      explanation.id = "preview-note";
      el("variant").closest("label").before(explanation);
      for (const id of ["variant", "boost", "spelling"]) {
        el(id).disabled = true;
        el(id).setAttribute("aria-describedby", explanation.id);
      }
      const previewOption = make("option", "", "Example word list");
      previewOption.value = "preview";
      el("variant").append(previewOption);
      el("variant").value = "preview";
      el("spelling").checked = false;
      el("boost-value").textContent = "—";
    } else el("boost-value").textContent = Number(el("boost").value).toFixed(2);
    drawKeyboard();
    setTab("compose");
    renderMode();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
