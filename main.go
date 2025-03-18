package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/pablobfonseca/go-vector/database"
	"github.com/pablobfonseca/go-vector/models"
	"github.com/pablobfonseca/go-vector/ollama"
	"github.com/pgvector/pgvector-go"
	"github.com/rs/cors"
	"github.com/spf13/viper"
)

func InsertTextEmbedding(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Text string `json:"text"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}

	embedding, err := ollama.GenerateEmbedding(req.Text)
	if err != nil {
		http.Error(w, "Failed to generate embedding", http.StatusInternalServerError)
		return
	}

	textEmbedding := models.TextEmbedding{
		Text:      req.Text,
		Embedding: pgvector.NewVector(embedding),
	}

	if err := database.DB.Create(&textEmbedding).Error; err != nil {
		http.Error(w, "Failed to insert", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	fmt.Fprintf(w, "Text embedded successfully!")
}

func SearchSimilarText(w http.ResponseWriter, r *http.Request) {
	var req struct {
		QueryText string `json:"query_text"`
		TopK      int    `json:"top_k"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}

	queryEmbedding, err := ollama.GenerateEmbedding(req.QueryText)
	if err != nil {
		http.Error(w, "Failed to generate embedding", http.StatusInternalServerError)
		return
	}

	var results []models.TextEmbedding
	err = database.DB.Raw(`
		SELECT * FROM text_embeddings
		ORDER BY embedding <-> ?
		LIMIT ?
		`, pgvector.NewVector(queryEmbedding), req.TopK).Scan(&results).Error

	if err != nil {
		http.Error(w, "Search failed", http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(results)
}

func main() {
	database.Connect()
	database.DB.AutoMigrate(&models.TextEmbedding{})

	r := mux.NewRouter()

	r.HandleFunc("/insert", InsertTextEmbedding).Methods("POST")
	r.HandleFunc("/search", SearchSimilarText).Methods("POST")

	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
	})

	handler := c.Handler(r)

	fmt.Println("Server running on port 8080...")
	log.Fatal(http.ListenAndServe(":8080", handler))
}

func init() {
	viper.SetConfigFile(".env")
	viper.SetConfigType("env")

	if err := viper.ReadInConfig(); err != nil {
		log.Println("Warning: Error reading .env file:", err)
	}
}
