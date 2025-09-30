// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

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