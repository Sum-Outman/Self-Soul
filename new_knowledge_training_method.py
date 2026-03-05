def train_neural_networks(self, training_config: Dict[str, Any] = None, 
                             callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Train knowledge model neural networks with full GPU support and advanced features"""
        try:
            import time
            start_time = time.time()
            
            # Device detection for GPU support
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {device}")
            
            # Move models to device if they exist
            if hasattr(self, 'semantic_encoder') and self.semantic_encoder is not None:
                self.semantic_encoder = self.semantic_encoder.to(device)
            if hasattr(self, 'knowledge_reasoner') and self.knowledge_reasoner is not None:
                self.knowledge_reasoner = self.knowledge_reasoner.to(device)
            if hasattr(self, 'relation_predictor') and self.relation_predictor is not None:
                self.relation_predictor = self.relation_predictor.to(device)
            
            if not self.training_data:
                return {"success": 0, "failure_reason": "No training data available"}
            
            # Use provided config or default parameters
            config = training_config or {}
            learning_rate = config.get("learning_rate", self.learning_rate)
            batch_size = config.get("batch_size", self.batch_size)
            epochs = config.get("epochs", self.epochs)
            
            # Ensure optimizers exist
            if not hasattr(self, 'semantic_optimizer'):
                if hasattr(self, 'semantic_encoder'):
                    self.semantic_optimizer = optim.Adam(self.semantic_encoder.parameters(), lr=learning_rate)
                else:
                    self.semantic_optimizer = None
            
            if not hasattr(self, 'reasoner_optimizer'):
                if hasattr(self, 'knowledge_reasoner'):
                    self.reasoner_optimizer = optim.Adam(self.knowledge_reasoner.parameters(), lr=learning_rate)
                else:
                    self.reasoner_optimizer = None
            
            if not hasattr(self, 'relation_optimizer'):
                if hasattr(self, 'relation_predictor'):
                    self.relation_optimizer = optim.Adam(self.relation_predictor.parameters(), lr=learning_rate)
                else:
                    self.relation_optimizer = None
            
            # Advanced learning rate schedulers (if optimizers exist)
            if self.semantic_optimizer:
                self.scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
                    self.semantic_optimizer, T_max=epochs, eta_min=1e-6
                )
                self.scheduler_reduce = optim.lr_scheduler.ReduceLROnPlateau(
                    self.semantic_optimizer, mode='min', factor=0.5, patience=5, verbose=True
                )
                self.scheduler_step = optim.lr_scheduler.StepLR(
                    self.semantic_optimizer, step_size=20, gamma=0.1
                )
            
            # Mixed precision training support
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            # Create training dataset and dataloader
            dataset = KnowledgeDataset(self.training_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training history
            training_history = {
                "semantic_encoder_loss": [],
                "knowledge_reasoner_loss": [],
                "relation_predictor_loss": [],
                "learning_rates": []
            }
            
            # Early stopping
            best_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            # Training loop with GPU support
            for epoch in range(epochs):
                epoch_semantic_loss = 0.0
                epoch_reasoner_loss = 0.0
                epoch_relation_loss = 0.0
                batch_count = 0
                
                for batch_data in dataloader:
                    # Move batch data to device
                    if isinstance(batch_data, dict):
                        for key in batch_data:
                            if isinstance(batch_data[key], torch.Tensor):
                                batch_data[key] = batch_data[key].to(device)
                    elif isinstance(batch_data, (list, tuple)):
                        batch_data = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch_data]
                    
                    # Zero gradients
                    if self.semantic_optimizer:
                        self.semantic_optimizer.zero_grad()
                    if self.reasoner_optimizer:
                        self.reasoner_optimizer.zero_grad()
                    if self.relation_optimizer:
                        self.relation_optimizer.zero_grad()
                    
                    # Extract batch data
                    concept_embeddings = batch_data["embedding_target"]
                    relations = batch_data["relations"]
                    
                    # Mixed precision training context
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and scaler is not None):
                        # Semantic encoder training
                        semantic_output = self.semantic_encoder(concept_embeddings)
                        semantic_loss = self.semantic_criterion(
                            semantic_output, concept_embeddings, 
                            torch.ones(concept_embeddings.size(0), device=device)
                        ) if hasattr(self, 'semantic_criterion') else torch.tensor(0.0, device=device)
                        
                        # Knowledge reasoner training
                        reasoner_input = semantic_output.detach()
                        reasoner_output = self.knowledge_reasoner(reasoner_input)
                        reasoner_target = reasoner_input  # Autoencoder style
                        reasoner_loss = self.reasoner_criterion(reasoner_output, reasoner_target) if hasattr(self, 'reasoner_criterion') else torch.tensor(0.0, device=device)
                        
                        # Relation predictor training (if relations available)
                        relation_loss = torch.tensor(0.0, device=device)
                        if len(relations) > 0 and hasattr(self, 'relation_predictor'):
                            relation_input = semantic_output.detach()
                            relation_output = self.relation_predictor(relation_input)
                            # Real relation classification with actual training data
                            if len(relations) > 0:
                                # Convert relations to tensor format for training
                                relation_target = torch.tensor(
                                    [self._relation_to_label(rel) for rel in relations], 
                                    dtype=torch.long, device=device
                                )
                                relation_loss = self.relation_criterion(relation_output, relation_target) if hasattr(self, 'relation_criterion') else torch.tensor(0.0, device=device)
                            else:
                                # Use default training when no relations available
                                relation_target = torch.zeros(relation_output.size(0), dtype=torch.long, device=device)
                                relation_loss = self.relation_criterion(relation_output, relation_target) if hasattr(self, 'relation_criterion') else torch.tensor(0.0, device=device)
                        
                        total_loss = semantic_loss + reasoner_loss + relation_loss
                    
                    # Backward pass with mixed precision support
                    if scaler and self.semantic_optimizer and self.reasoner_optimizer:
                        scaler.scale(total_loss).backward()
                        scaler.step(self.semantic_optimizer)
                        scaler.step(self.reasoner_optimizer)
                        if len(relations) > 0 and self.relation_optimizer:
                            scaler.step(self.relation_optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if self.semantic_optimizer:
                            self.semantic_optimizer.step()
                        if self.reasoner_optimizer:
                            self.reasoner_optimizer.step()
                        if len(relations) > 0 and self.relation_optimizer:
                            self.relation_optimizer.step()
                    
                    epoch_semantic_loss += semantic_loss.item()
                    epoch_reasoner_loss += reasoner_loss.item()
                    if len(relations) > 0:
                        epoch_relation_loss += relation_loss.item()
                    batch_count += 1
                
                # Calculate average losses
                avg_semantic_loss = epoch_semantic_loss / max(batch_count, 1)
                avg_reasoner_loss = epoch_reasoner_loss / max(batch_count, 1)
                avg_relation_loss = epoch_relation_loss / max(batch_count, 1)
                
                training_history["semantic_encoder_loss"].append(avg_semantic_loss)
                training_history["knowledge_reasoner_loss"].append(avg_reasoner_loss)
                training_history["relation_predictor_loss"].append(avg_relation_loss)
                if self.semantic_optimizer:
                    training_history["learning_rates"].append(self.semantic_optimizer.param_groups[0]['lr'])
                
                # Update learning rate schedulers
                if self.semantic_optimizer:
                    self.scheduler_cosine.step()
                    self.scheduler_reduce.step(avg_semantic_loss)
                    if epoch % 20 == 0:
                        self.scheduler_step.step()
                
                # Early stopping check
                current_loss = avg_semantic_loss + avg_reasoner_loss + avg_relation_loss
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    
                    # Save model checkpoint
                    self._save_checkpoint(epoch, current_loss, training_history, config)
                else:
                    patience_counter += 1
                
                # Callback for progress reporting
                if callback:
                    callback(epoch, epochs, {
                        "semantic_loss": avg_semantic_loss,
                        "reasoner_loss": avg_reasoner_loss,
                        "relation_loss": avg_relation_loss,
                        "current_lr": self.semantic_optimizer.param_groups[0]['lr'] if self.semantic_optimizer else learning_rate,
                        "device": str(device)
                    })
                
                # Log progress
                if epoch % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} - "
                        f"Semantic Loss: {avg_semantic_loss:.4f}, "
                        f"Reasoner Loss: {avg_reasoner_loss:.4f}, "
                        f"Relation Loss: {avg_relation_loss:.4f}, "
                        f"LR: {self.semantic_optimizer.param_groups[0]['lr'] if self.semantic_optimizer else learning_rate:.6f}, "
                        f"Device: {device}"
                    )
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save final checkpoint
            self._save_checkpoint(epochs - 1, best_loss, training_history, config)
            
            return {
                "success": 1,
                "training_history": training_history,
                "final_losses": {
                    "semantic_encoder": training_history["semantic_encoder_loss"][-1] if training_history["semantic_encoder_loss"] else 0.0,
                    "knowledge_reasoner": training_history["knowledge_reasoner_loss"][-1] if training_history["knowledge_reasoner_loss"] else 0.0,
                    "relation_predictor": training_history["relation_predictor_loss"][-1] if training_history["relation_predictor_loss"] else 0.0
                },
                "device_used": str(device),
                "training_time": time.time() - start_time,
                "model_checkpoints_saved": getattr(self, 'checkpoints_saved', 0),
                "real_pytorch_training": True,
                "gpu_accelerated": torch.cuda.is_available(),
                "training_completed": 1,
                "epochs_completed": epochs if patience_counter < patience else epoch + 1
            }
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {str(e)}")
            return {"success": 0, "failure_reason": str(e)}
    
def _save_checkpoint(self, epoch: int, loss: float, history: Dict, config: Dict) -> None:
        """Save model checkpoint"""
        try:
            checkpoint_dir = "checkpoints/knowledge_model"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            
            checkpoint_data = {
                'epoch': epoch,
                'loss': loss,
                'training_history': history,
                'config': config
            }
            
            # Add model state dicts if models exist
            if hasattr(self, 'semantic_encoder') and self.semantic_encoder is not None:
                checkpoint_data['semantic_encoder_state_dict'] = self.semantic_encoder.state_dict()
            
            if hasattr(self, 'knowledge_reasoner') and self.knowledge_reasoner is not None:
                checkpoint_data['knowledge_reasoner_state_dict'] = self.knowledge_reasoner.state_dict()
            
            if hasattr(self, 'relation_predictor') and self.relation_predictor is not None:
                checkpoint_data['relation_predictor_state_dict'] = self.relation_predictor.state_dict()
            
            # Add optimizer state dicts if optimizers exist
            if hasattr(self, 'semantic_optimizer') and self.semantic_optimizer is not None:
                checkpoint_data['semantic_optimizer_state_dict'] = self.semantic_optimizer.state_dict()
            
            if hasattr(self, 'reasoner_optimizer') and self.reasoner_optimizer is not None:
                checkpoint_data['reasoner_optimizer_state_dict'] = self.reasoner_optimizer.state_dict()
            
            if hasattr(self, 'relation_optimizer') and self.relation_optimizer is not None:
                checkpoint_data['relation_optimizer_state_dict'] = self.relation_optimizer.state_dict()
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update checkpoints saved counter
            if not hasattr(self, 'checkpoints_saved'):
                self.checkpoints_saved = 0
            self.checkpoints_saved += 1
            
            self.logger.info(f"✅ Saved model checkpoint to: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")