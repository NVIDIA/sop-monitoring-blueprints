// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React from 'react';
import { Card, Button, Row, Col, ListGroup, Badge } from 'react-bootstrap';

// Use nginx proxy path to avoid CORS issues
const API_BASE_URL = '/api/annotation';

const ClipsList = ({ clips }) => {
  if (!clips || clips.length === 0) {
    return null;
  }

  return (
    <Card className="mt-4">
      <Card.Header>
        <h4>Split Results ({clips.length} segments)</h4>
      </Card.Header>
      <ListGroup variant="flush">
        {clips.map((clip, index) => (
          <ListGroup.Item key={clip.id} className="py-3">
            <Row>
              <Col md={7}>
                <h5>Segment {index + 1}: {clip.filename}</h5>
                <div className="text-muted">
                  <div>Start time: {parseFloat(clip.start_time).toFixed(2)}s</div>
                  <div>End time: {parseFloat(clip.end_time).toFixed(2)}s</div>
                  <div>Duration: {parseFloat(clip.duration).toFixed(2)}s</div>
                </div>
              </Col>
              <Col md={5} className="d-flex align-items-center">
                <div className="ms-auto">
                  <a
                    href={`${API_BASE_URL}/api/v1/videos/${clip.id}/download`}
                    className="btn btn-primary me-2"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download
                  </a>
                  <a
                    href={`${API_BASE_URL}/api/v1/videos/${clip.id}`}
                    className="btn btn-outline-secondary"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Details
                  </a>
                </div>
              </Col>
            </Row>
          </ListGroup.Item>
        ))}
      </ListGroup>
      <Card.Footer>
        <Button
          variant="success"
          href={`${API_BASE_URL}/api/v1/videos`}
          target="_blank"
        >
          View All Videos
        </Button>
      </Card.Footer>
    </Card>
  );
};

export default ClipsList;