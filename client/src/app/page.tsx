"use client";
import axios from "axios";
import Image from "next/image";
import { useState } from "react";

type Embedding = {
  id: string;
  text: string;
  embedding: Array<number>;
};

export default function Home() {
  const [text, setText] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [results, setResults] = useState<Array<Embedding>>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleInsert = async () => {
    setLoading(true);
    await axios.post("http://localhost:8080/insert", { text });
    setText("");
    setLoading(false);
  };

  const handleSearch = async () => {
    setLoading(true);
    const res = await axios.post("http://localhost:8080/search", {
      query_text: query,
      top_k: 5,
    });
    setResults(res.data);
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-4">
      <h1 className="text-3xl font-bold mb-4">🔍 PGVector + Ollama</h1>

      <div className="mb-4">
        <input
          type="text"
          className="p-2 rounded border border-gray-600 text-white bg-gray-700"
          placeholder="Enter text to insert"
          required
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          onClick={handleInsert}
          className="ml-2 px-4 py-2 bg-blue-500 rounded hover:bg-blue-600"
        >
          Insert
        </button>
      </div>

      <div className="mb-4">
        <input
          type="text"
          className="p-2 rounded border border-gray-600 bg-gray-700 text-white"
          value={query}
          required
          placeholder="Search for similar texts"
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          onClick={handleSearch}
          className="ml-2 px-4 py-2 bg-green-500 rounded hover:bg-green-600"
        >
          Search
        </button>
      </div>

      {loading && <p>Loading...</p>}

      <div className="mt-4">
        {results.map((item, i) => (
          <div key={i} className="p-2 bg-gray-800 rounded mb-2">
            {item.text}
          </div>
        ))}
      </div>
    </div>
  );
}
